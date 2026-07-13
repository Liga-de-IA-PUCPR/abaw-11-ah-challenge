"""Entrypoint Hydra do desafio BAH (áudio + texto · síntese).

`@hydra.main` compõe a config a partir dos grupos (README §7) e faz *dispatch*
por ``cfg.mode``:

    preprocess   índice (FASE 2) + extração de áudio (mp4→flac 16 kHz) + janelas → cache
    featurize    janelas → embedders (texto+áudio) + tabular → Parquet (FASE 3)
    train        treina o modelo (RF sklearn/CPU OU cross_attention Lightning/MPS) (FASE 4)
    evaluate     avalia a nível de vídeo (Macro-F1, AP) em um split (FASE 4/5)
    submit       escreve o arquivo de submissão (video_id, pred) (FASE 5)

Reusa a estrutura do ``main.py`` da branch ``matheus`` (Hydra + WandbLogger +
L.Trainer), mas **generalizada para os 2 modelos e os 5 modos**: o modelo e o
trainer vêm das *factories* da FASE 4 (``create_model`` + ``create_trainer``),
de modo que o caminho ``random_forest`` **nunca importa Lightning**.

Exemplos::

    # baseline RandomForest (CPU, sem Lightning) — usa todos os defaults
    python main.py

    # pré-processamento e featurização (embeddings em Apple Metal)
    python main.py mode=preprocess
    python main.py mode=featurize device=mps

    # modelo de competição (preset cross_attention) em MPS
    PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py +experiment=cross_attention device=mps

    # GNN heterogêneo (gnn-modalblocks) — ver references/gnn_training_procedure.md
    uv run python main.py +experiment=hetero_gnn mode=train device=cuda

    # MULTIRUN (sweep paralelo via joblib launcher)
    python main.py -m model.lr=1e-3,5e-4 model.num_heads=4,8 data.window.size_s=4,5,6
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from src.logger import get_logger

log = get_logger("main")


# ==============================================================================
# Dispatch por modo — cada handler importa só o que precisa (imports tardios)
# ==============================================================================


def _run_preprocess(cfg: DictConfig) -> int:
    """``mode=preprocess`` — índice + extração de áudio + janelas + cache (FASE 2)."""
    from src.pipeline.preprocess import run_preprocess

    summary = run_preprocess(cfg)
    log.info(
        f"Preprocess concluído: {summary['n_videos']} vídeos, "
        f"{summary['n_windows']} janelas; áudio em {summary['audio_dir']}; "
        f"índice em {summary['windows_index']}."
    )
    return 0


def _run_featurize(cfg: DictConfig, device) -> int:
    """``mode=featurize`` — janelas → embedders → tabular → Parquet (FASE 3)."""
    from src.pipeline.featurize import run_featurize

    log.info(f"Device dos embedders: {device}")
    summary = run_featurize(cfg, device=device)
    if summary.get("cached"):
        log.info(
            f"Featurize: cache reutilizado — {summary['n_windows']} janelas em {summary['parquet_path']}."
        )
    else:
        log.info(
            f"Featurize concluído: {summary['n_windows']} janelas → {summary['parquet_path']} "
            f"(d_text={summary['d_text']}, d_audio={summary['d_audio']}, d_tab={summary['d_tab']})."
        )
    return 0


def _as_loader(cfg: DictConfig, data, family: str, split: str):
    """Adapta ``data`` ao que o trainer da ``family`` espera (FASE 6).

    O caminho ``sklearn`` consome ``WindowMatrixView`` direto (CPU, sem torch) →
    devolve ``data`` inalterado. O caminho ``lightning`` precisa de um
    ``DataLoader`` com ``collate_fn=collate_sequences`` (padding até T_max +
    ``key_padding_mask``); sem ele o ``LightningTrainer`` recebe um ``Dataset``
    cru (``_labels_from_loader`` quebra em ``loader.dataset`` e o
    ``_shared_step`` quebra na chave ausente ``key_padding_mask``).

    O import do ``torch``/``collate_sequences`` é **lazy** para preservar o
    caminho ``sklearn`` 100% sem torch (README §7).
    """
    if family != "lightning":
        return data

    from torch.utils.data import DataLoader

    from src.data.datasets import collate_sequences

    sampler = None
    shuffle = split == "train"
    hard_path = cfg.data.get("hard_examples") if hasattr(cfg.data, "get") else None
    if split == "train" and hard_path:
        sampler = _build_weighted_sampler(hard_path, data)
        if sampler is not None:
            shuffle = False  # sampler e shuffle são mutuamente exclusivos

    return DataLoader(
        data,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_sequences,
    )


def _build_weighted_sampler(hard_path: str, dataset):
    """WeightedRandomSampler alinhado à ordem de ``dataset.video_ids`` (mode=hard_mining)."""
    import json
    from pathlib import Path

    p = Path(hard_path)
    if not p.exists():
        log.warning(f"hard_examples ausente ({p}); amostragem uniforme.")
        return None

    from torch.utils.data import WeightedRandomSampler

    payload = json.loads(p.read_text(encoding="utf-8"))
    weights_map = payload.get("weights", payload)
    video_ids = getattr(dataset, "video_ids", None)
    if not video_ids:
        log.warning("Dataset sem video_ids; amostragem uniforme.")
        return None

    weights = [float(weights_map.get(str(vid), 1.0)) for vid in video_ids]
    log.info(
        f"WeightedRandomSampler: {len(weights)} amostras, "
        f"peso∈[{min(weights):.2f}, {max(weights):.2f}] de {p}"
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _build_trainer(cfg: DictConfig, device):
    """Constrói modelo (FASE 4) + trainer (FASE 4) a partir das *factories*.

    ``create_model`` devolve ``(model, family)``; ``create_trainer`` escolhe o
    ``SklearnTrainer`` (family="sklearn", CPU) ou o ``LightningTrainer``
    (family="lightning", mapeia device→accelerator). O import do Lightning
    acontece **lazy** dentro de ``create_trainer`` só quando family=="lightning".

    O device **não** é passado à factory: os trainers o resolvem internamente
    via ``resolve_device(cfg.device)`` (README §7). O ``device`` aqui serve
    apenas para log/coerência com os demais handlers.
    """
    from src.models.registry import create_model
    from src.training.factory import create_trainer

    model, family = create_model(cfg.model.name, cfg.model)
    log.info(f"Modelo='{cfg.model.name}' (family={family}, device={device}).")
    trainer = create_trainer(family, model=model, cfg=cfg)
    return trainer, family


def _run_train(cfg: DictConfig, device) -> int:
    """``mode=train`` — treina + calibra limiar + salva checkpoint (FASE 4/5)."""
    import src.models  # noqa: F401  — dispara os decorators @register_model
    from src.data.datasets import load_train_val
    from src.outputs.checkpoint import resolve_output_dir

    trainer, family = _build_trainer(cfg, device)

    # Carrega o cache Parquet (FASE 3) na visão certa por família (FASE 2):
    #   sklearn   → WindowMatrixView  (X achatado por janela)
    #   lightning → VideoSequenceDataset (sequência (T, D) por vídeo)
    train_data, val_data = load_train_val(cfg, family=family)
    # Lightning precisa de DataLoaders (collate_sequences → padding + mask); o
    # caminho sklearn passa a WindowMatrixView inalterada.
    train_data = _as_loader(cfg, train_data, family, "train")
    val_data = _as_loader(cfg, val_data, family, "val")

    # Resolve o run dir ANTES do fit: o ModelCheckpoint do Lightning grava o .ckpt
    # DENTRO deste dir (junto do trainer_state.json), então evaluate/submit resolvem
    # UM só diretório. O SklearnTrainer ignora self.output_dir.
    out_dir = resolve_output_dir(cfg.data.paths.output_root, cfg.model.name)
    trainer.output_dir = str(out_dir)

    result = trainer.fit(train_data, val_data)
    log.info(
        f"Treino concluído (family={family}). "
        f"Macro-F1(val)={result['val']['macro_f1']:.4f} | "
        f"AP(val)={result['val'].get('average_precision', 0.0):.4f} | "
        f"limiar={result.get('threshold', 0.5):.4f}."
    )

    trainer.save(out_dir)
    log.info(f"Checkpoint salvo em: {out_dir}")
    return 0


def _apply_threshold_override(cfg: DictConfig, trainer) -> None:
    """Sobrepõe o limiar salvo no checkpoint por ``aggregation.threshold`` (se float).

    Permite RE-AVALIAR/submeter com um limiar diferente SEM re-treinar: ``"auto"``
    (default) mantém o limiar calibrado no ``fit``; um float o sobrepõe na hora.
    Vale para as duas famílias (RF e cross_attention).
    """
    agg = getattr(cfg, "aggregation", None)
    thr = agg.get("threshold", "auto") if agg is not None else "auto"
    if thr not in (None, "auto"):
        trainer.threshold_ = float(thr)
        log.info(f"Limiar sobreposto pela config (sem re-treinar): {trainer.threshold_:.4f}")


def _write_eval_report(cfg: DictConfig, trainer, data, report, split: str, ckpt_dir) -> None:
    """Gera metrics.json + results.txt + plots automaticamente após CADA evaluate (FASE 5).

    Salva em ``<ckpt_dir>/eval_<split>/``. Tudo degrada graciosamente — um plot sem
    insumo é pulado com aviso, nunca derruba o evaluate.
    """
    import json
    from pathlib import Path

    import numpy as np

    from src.outputs.reporter import Reporter, load_video_metadata_for_eval
    from src.training.aggregation import threshold_curve

    out_dir = Path(ckpt_dir) / f"eval_{split}"
    rep = Reporter(out_dir)
    threshold = float(getattr(trainer, "threshold_", 0.5) or 0.5)
    method = getattr(trainer, "method", "identity")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    rep.save_metrics_json(report, cfg_dict, threshold=threshold, aggregation_method=method)
    rep.save_results_txt({**report, "threshold": threshold, "aggregation_method": method})

    # Plots a nível de vídeo (confusão, PR, curva de limiar) — exige video_outputs.
    if hasattr(trainer, "video_outputs"):
        try:
            o = trainer.video_outputs(data)
            rep.plot_confusion_matrix(o["y_true"], o["y_pred"])
            rep.plot_roc_curve(o["y_true"], o["y_proba"])
            rep.plot_precision_recall(o["y_true"], o["y_proba"])
            grid, f1s = threshold_curve(o["y_true"], o["y_proba"])
            rep.plot_threshold_curve(grid, f1s, threshold)
            meta = load_video_metadata_for_eval(cfg, o["video_ids"])
            rep.save_predictions_csv(
                o["video_ids"],
                o["y_true"],
                o["y_proba"],
                o["y_pred"],
                threshold,
                metadata=meta,
            )
            rep.save_error_analysis(
                o["video_ids"],
                o["y_true"],
                o["y_pred"],
                metadata=meta,
            )
        except Exception as exc:  # noqa: BLE001 — plots nunca derrubam o evaluate
            log.warning(f"Plots a nível de vídeo pulados: {exc}")

    # Importâncias de features (RandomForest): individual + AGRUPADO (texto/áudio como
    # 1 feature cada vs tabulares). Nomes/grupos vêm do sidecar do Parquet (FASE 3).
    model = getattr(trainer, "model", None)
    imp = (
        model.feature_importances()
        if model is not None and hasattr(model, "feature_importances")
        else None
    )
    if imp is not None:
        imp = np.asarray(imp)
        fn: dict[str, list[str]] = {}
        sidecar = Path(cfg.data.paths.parquet_path).with_suffix(".json")
        if sidecar.exists():
            fn = json.loads(sidecar.read_text(encoding="utf-8")).get("feature_names", {})
        n_audio, n_text, tab = (
            len(fn.get("audio", [])),
            len(fn.get("text", [])),
            list(fn.get("tabular", [])),
        )
        names = getattr(model, "feature_names", None) or (
            list(fn.get("audio", [])) + list(fn.get("text", [])) + tab
        )
        if names and len(names) == len(imp):
            rep.plot_feature_importances(imp, list(names))
        # Panorama: texto/áudio agregados (1 barra cada) vs cada feature tabular do dataset.
        if n_audio and n_text and (n_audio + n_text + len(tab)) == len(imp):
            group_of = ["áudio (emb)"] * n_audio + ["texto (emb)"] * n_text + tab
            rep.plot_grouped_feature_importances(imp, group_of)

    log.info(f"Relatórios + plots salvos em: {out_dir}")


def _run_evaluate(cfg: DictConfig, device) -> int:
    """``mode=evaluate`` — métricas a nível de vídeo num split (FASE 4/5)."""
    import json

    from src.data.datasets import load_split
    from src.outputs.checkpoint import resolve_latest_checkpoint
    from src.training.factory import load_trainer

    _, family = _build_trainer(cfg, device)
    # resolve_latest_checkpoint filtra pela família (FASE 5): só os artefatos do
    # modelo atual (model.joblib p/ RF | *.ckpt p/ neural) → devolve o mais recente.
    ckpt_dir = cfg.get("checkpoint") or resolve_latest_checkpoint(
        cfg.data.paths.output_root, family=family
    )
    trainer = load_trainer(family, ckpt_dir, cfg=cfg)
    _apply_threshold_override(cfg, trainer)  # aggregation.threshold=<float> sobrepõe o salvo
    log.info(f"Checkpoint carregado: {ckpt_dir}")

    split = cfg.get("split") or "val"
    data = _as_loader(cfg, load_split(cfg, split, family=family), family, split)
    report = trainer.evaluate(data)
    log.info(
        f"[{split}] Macro-F1={report['macro_f1']:.4f} | "
        f"AP={report.get('average_precision', 0.0):.4f} (n={report['n_videos']})."
    )
    _write_eval_report(cfg, trainer, data, report, split, ckpt_dir)  # plots + relatórios (FASE 5)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _run_submit(cfg: DictConfig, device) -> int:
    """``mode=submit`` — escreve o arquivo de submissão (video_id, pred) (FASE 5)."""
    from pathlib import Path

    from src.data.datasets import load_split
    from src.outputs.checkpoint import resolve_latest_checkpoint
    from src.outputs.submission import write_submission
    from src.training.factory import load_trainer

    _, family = _build_trainer(cfg, device)
    # resolve_latest_checkpoint filtra pela família do modelo atual (FASE 5).
    ckpt_dir = cfg.get("checkpoint") or resolve_latest_checkpoint(
        cfg.data.paths.output_root, family=family
    )
    trainer = load_trainer(family, ckpt_dir, cfg=cfg)
    _apply_threshold_override(cfg, trainer)  # aggregation.threshold=<float> sobrepõe o salvo

    split = cfg.get("split") or "test"
    out_path = Path(cfg.get("out") or "outputs/submission.txt")
    data = _as_loader(cfg, load_split(cfg, split, family=family), family, split)
    video_preds = trainer.predict(data)  # {video_id: 0/1}
    write_submission(video_preds, out_path)
    log.info(f"Submissão escrita: {out_path} ({len(video_preds)} vídeos).")
    return 0


def _run_pretrain_gae(cfg: DictConfig, device) -> int:
    """``mode=pretrain_gae`` — pré-treino HeteroGAE (recon_loss, sem labels)."""
    from src.data.datasets import load_train_val
    from src.models.hetero_gae_pretrain import HeteroGaePretrain
    from src.outputs.checkpoint import resolve_output_dir
    from src.training.gae_pretrain_trainer import GaePretrainTrainer

    model = HeteroGaePretrain.from_config(cfg.model)
    trainer = GaePretrainTrainer(model=model, config=cfg)
    train_data, val_data = load_train_val(cfg, family="lightning")
    train_data = _as_loader(cfg, train_data, "lightning", "train")
    val_data = _as_loader(cfg, val_data, "lightning", "val")

    out_dir = resolve_output_dir(cfg.data.paths.output_root, cfg.model.name)
    trainer.output_dir = str(out_dir)
    trainer.fit(train_data, val_data)
    trainer.save(out_dir)
    log.info(f"Pré-treino GAE concluído. Encoder: {out_dir / 'gae_encoder.pt'}")
    return 0


def _ensemble_member_video_scores(
    cfg: DictConfig,
    member,
    split: str,
    device,
) -> dict[str, float]:
    """Scores contínuos por vídeo de um membro do ensemble (sklearn ou lightning)."""
    from pathlib import Path

    from omegaconf import OmegaConf

    from src.data.datasets import load_split
    from src.models.registry import get_family
    from src.outputs.checkpoint import resolve_latest_checkpoint
    from src.training.aggregation import video_scores
    from src.training.factory import load_trainer

    model_name = str(member.model)
    member_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model_yaml = Path("configs/model") / f"{model_name}.yaml"
    if not model_yaml.exists():
        raise FileNotFoundError(f"YAML ausente: {model_yaml}")
    member_cfg.model = OmegaConf.load(str(model_yaml))

    ckpt = member.get("checkpoint")
    if ckpt in (None, "null"):
        family = get_family(model_name)
        ckpt_dir = resolve_latest_checkpoint(
            cfg.data.paths.output_root,
            family=family,
            model_name=model_name,
        )
    else:
        ckpt_dir = ckpt
        family = get_family(model_name)

    trainer = load_trainer(family, ckpt_dir, cfg=member_cfg)

    if family == "sklearn":
        data_raw = load_split(cfg, split, family="sklearn")
        proba = trainer.model.predict_proba(data_raw.X)
        raw = video_scores(proba, data_raw.video_ids, trainer.method)
        if hasattr(trainer, "_calibrated_scores_dict"):
            return trainer._calibrated_scores_dict(raw)
        return raw

    data_raw = load_split(cfg, split, family="lightning")
    data = _as_loader(cfg, data_raw, "lightning", split)
    ids, proba = trainer._infer(data)
    return {str(v): float(p) for v, p in zip(ids, proba, strict=False)}


def _ensemble_infer_split(
    cfg: DictConfig,
    ensemble,
    split: str,
    device,
) -> tuple[list[dict[str, float]], list[float], dict[str, int]]:
    """Inferência de todos os membros do ensemble num split; retorna scores, pesos e labels."""
    from src.data.datasets import load_split

    labels_raw = load_split(cfg, split, family="lightning")
    labels = {str(vid): int(lab) for vid, lab in labels_raw.video_labels.items()}

    member_scores: list[dict[str, float]] = []
    weights: list[float] = []
    for member in ensemble.members:
        model_name = str(member.model)
        try:
            scores = _ensemble_member_video_scores(cfg, member, split, device)
        except FileNotFoundError as exc:
            log.warning(f"Ensemble: membro '{model_name}' ignorado — {exc}")
            continue
        member_scores.append(scores)
        weights.append(float(member.get("weight", 1.0)))
        log.info(f"Ensemble member '{model_name}' carregado")
    return member_scores, weights, labels


def _ensemble_combine_scores(
    member_scores: list[dict[str, float]],
    weights: list[float],
    combine: str,
) -> dict[str, float]:
    import numpy as np

    video_ids = sorted(set().union(*member_scores))
    combined: dict[str, float] = {}
    for vid in video_ids:
        vals = []
        ws = []
        for scores, w in zip(member_scores, weights, strict=True):
            if vid in scores:
                vals.append(scores[vid])
                ws.append(w)
        if not vals:
            continue
        if combine == "weighted":
            combined[vid] = float(np.average(vals, weights=ws))
        else:
            combined[vid] = float(np.mean(vals))
    return combined


def _run_ensemble_evaluate(cfg: DictConfig, device) -> int:
    """``mode=ensemble_evaluate`` — combina probas de N checkpoints e calibra limiar."""
    import json
    from pathlib import Path

    import numpy as np

    from src.training.aggregation import calibrate_threshold
    from src.training.metrics import evaluate_video_predictions

    ensemble = cfg.get("ensemble")
    if ensemble is None:
        raise ValueError("mode=ensemble_evaluate requer +ensemble=default ou bloco ensemble na config.")

    split = cfg.get("split") or "val"
    combine = str(ensemble.get("combine", "mean"))
    agg = getattr(cfg, "aggregation", {}) or {}
    thr_setting = agg.get("threshold", "auto")

    # Limiar SEMPRE calibrado na val (nunca no split de avaliação).
    member_scores_val, weights, val_labels = _ensemble_infer_split(cfg, ensemble, "val", device)
    if not member_scores_val:
        raise FileNotFoundError(
            "Ensemble: nenhum membro com checkpoint válido. Treine ao menos um modelo."
        )
    val_combined = _ensemble_combine_scores(member_scores_val, weights, combine)

    if thr_setting == "auto":
        val_ids = np.array([v for v in val_combined if v in val_labels])
        val_proba = np.array([val_combined[v] for v in val_ids], dtype=np.float32)
        threshold, val_f1 = calibrate_threshold(
            val_proba=val_proba,
            val_video_ids=val_ids,
            val_video_labels=val_labels,
            method="identity",
            metric="macro_f1",
            selection=agg.get("calibration", "smooth"),
            smooth_window=float(agg.get("smooth_window", 0.10)),
            target_pos_rate=(
                None
                if agg.get("target_pos_rate") in (None, "null")
                else float(agg.get("target_pos_rate"))
            ),
        )
        log.info(f"Ensemble: limiar calibrado na val={threshold:.4f} (macro_f1={val_f1:.4f})")
    else:
        threshold = float(thr_setting)

    if split == "val":
        combined = val_combined
        labels = val_labels
    else:
        member_scores_eval, _, labels = _ensemble_infer_split(cfg, ensemble, split, device)
        combined = _ensemble_combine_scores(member_scores_eval, weights, combine)

    preds = {vid: int(combined[vid] >= threshold) for vid in combined if vid in labels}
    report = evaluate_video_predictions(
        video_labels=labels, video_pred=preds, video_score=combined
    )
    log.info(
        f"Ensemble [{split}] Macro-F1={report['macro_f1']:.4f} | "
        f"AP={report.get('average_precision', 0.0):.4f} | limiar={threshold:.4f}"
    )
    out_dir = Path(cfg.data.paths.output_root) / "ensemble_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"report_{split}.json").write_text(
        json.dumps({**report, "threshold": threshold, "combine": combine}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _run_ensemble_submit(cfg: DictConfig, device) -> int:
    """``mode=ensemble_submit`` — calibra limiar na val e gera submissão no teste."""
    from pathlib import Path

    import numpy as np

    from src.outputs.submission import write_submission
    from src.training.aggregation import calibrate_threshold

    ensemble = cfg.get("ensemble")
    if ensemble is None:
        raise ValueError("mode=ensemble_submit requer +ensemble=... na config.")

    combine = str(ensemble.get("combine", "mean"))
    member_scores_val, weights, val_labels = _ensemble_infer_split(cfg, ensemble, "val", device)
    if not member_scores_val:
        raise FileNotFoundError("Ensemble: nenhum membro válido para submit.")

    val_combined = _ensemble_combine_scores(member_scores_val, weights, combine)
    val_ids = np.array([v for v in val_combined if v in val_labels])
    val_proba = np.array([val_combined[v] for v in val_ids], dtype=np.float32)
    agg = getattr(cfg, "aggregation", {}) or {}
    threshold, val_f1 = calibrate_threshold(
        val_proba=val_proba,
        val_video_ids=val_ids,
        val_video_labels=val_labels,
        method="identity",
        metric="macro_f1",
        selection=agg.get("calibration", "smooth"),
        smooth_window=float(agg.get("smooth_window", 0.10)),
        target_pos_rate=(
            None
            if agg.get("target_pos_rate") in (None, "null")
            else float(agg.get("target_pos_rate"))
        ),
    )
    log.info(f"Ensemble submit: limiar val={threshold:.4f} (macro_f1={val_f1:.4f})")

    member_scores_test, _, _ = _ensemble_infer_split(cfg, ensemble, "test", device)
    test_combined = _ensemble_combine_scores(member_scores_test, weights, combine)
    preds = {vid: int(score >= threshold) for vid, score in test_combined.items()}
    out_path = Path(cfg.get("out") or "outputs/submission_ensemble.txt")
    write_submission(preds, out_path)
    log.info(f"Submissão ensemble: {out_path} ({len(preds)} vídeos).")
    return 0


def _run_hard_mining(cfg: DictConfig, device) -> int:
    """``mode=hard_mining`` — pontua o split de treino e gera pesos de amostragem.

    Roda o checkpoint atual sobre o **treino**, identifica *hard positives* (rótulo 1,
    proba baixa) e *hard negatives* (rótulo 0, proba alta) + casos limítrofes, e grava
    ``{video_id: weight}`` em JSON. O treino consome esses pesos via
    ``WeightedRandomSampler`` (``data.hard_examples=<json>``).
    """
    import json
    from pathlib import Path

    from src.data.datasets import load_split
    from src.outputs.checkpoint import resolve_latest_checkpoint
    from src.outputs.reporter import load_video_metadata_for_eval
    from src.training.factory import load_trainer

    _, family = _build_trainer(cfg, device)
    if family != "lightning":
        raise ValueError("mode=hard_mining requer um modelo da família lightning.")

    ckpt_dir = cfg.get("checkpoint") or resolve_latest_checkpoint(
        cfg.data.paths.output_root, family=family
    )
    trainer = load_trainer(family, ckpt_dir, cfg=cfg)
    log.info(f"Hard mining: checkpoint {ckpt_dir}")

    split = cfg.get("split") or "train"
    data = _as_loader(cfg, load_split(cfg, split, family=family), family, split)
    o = trainer.video_outputs(data)

    hard_hi = float(cfg.get("hard_hi", 0.7))
    hard_lo = float(cfg.get("hard_lo", 0.3))
    hard_weight = float(cfg.get("hard_weight", 3.0))
    border_margin = float(cfg.get("hard_border", 0.1))
    border_weight = float(cfg.get("hard_border_weight", 2.0))
    rare_types = set(cfg.get("hard_rare_types", ["neutral", "willing"]))
    rare_mult = float(cfg.get("hard_rare_mult", 1.5))

    meta = load_video_metadata_for_eval(cfg, o["video_ids"])
    threshold = float(getattr(trainer, "threshold_", 0.5) or 0.5)

    weights: dict[str, float] = {}
    n_hard = n_border = n_rare = 0
    for i, vid in enumerate(o["video_ids"]):
        vid = str(vid)
        yt = int(o["y_true"][i])
        p = float(o["y_proba"][i])
        w = 1.0
        is_hard_pos = yt == 1 and p < hard_lo
        is_hard_neg = yt == 0 and p > hard_hi
        if is_hard_pos or is_hard_neg:
            w *= hard_weight
            n_hard += 1
        elif abs(p - threshold) < border_margin:
            w *= border_weight
            n_border += 1
        qtype = (meta.get(vid, {}) or {}).get("question_type")
        if qtype in rare_types:
            w *= rare_mult
            n_rare += 1
        weights[vid] = round(w, 4)

    out_path = Path(cfg.get("out") or "data/interim/hard_examples.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "checkpoint": str(ckpt_dir),
        "threshold": threshold,
        "params": {
            "hard_hi": hard_hi,
            "hard_lo": hard_lo,
            "hard_weight": hard_weight,
            "border_margin": border_margin,
            "border_weight": border_weight,
            "rare_types": sorted(rare_types),
            "rare_mult": rare_mult,
        },
        "n_videos": len(weights),
        "n_hard": n_hard,
        "n_border": n_border,
        "n_rare": n_rare,
        "weights": weights,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        f"Hard mining: {len(weights)} vídeos → {out_path} "
        f"(hard={n_hard}, borderline={n_border}, raros={n_rare})."
    )
    return 0


def _run_featurize_face(cfg: DictConfig) -> int:
    """``mode=featurize_face`` — MediaPipe Face Mesh → coluna face_landmarks no Parquet."""
    from src.pipeline.featurize_face import run_featurize_face

    summary = run_featurize_face(cfg)
    log.info(
        f"Featurize face concluído: {summary.get('n_windows', '?')} janelas → "
        f"{summary['parquet_path']} (d_face={summary.get('d_face', '?')})."
    )
    return 0


_DISPATCH = {
    "preprocess": lambda cfg, dev: _run_preprocess(cfg),
    "featurize": _run_featurize,
    "featurize_face": lambda cfg, dev: _run_featurize_face(cfg),
    "train": _run_train,
    "pretrain_gae": _run_pretrain_gae,
    "evaluate": _run_evaluate,
    "ensemble_evaluate": _run_ensemble_evaluate,
    "ensemble_submit": _run_ensemble_submit,
    "hard_mining": _run_hard_mining,
    "submit": _run_submit,
}


# ==============================================================================
# Entrypoint Hydra
# ==============================================================================


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> int:
    """Ponto de entrada Hydra.

    Compõe a config (grupos + ``defaults`` list + presets ``experiment/*`` +
    overrides de CLI + multirun via launcher joblib), fixa a seed, resolve o
    device e despacha para o handler do ``cfg.mode``.

    Args:
        cfg: Config composta pelo Hydra (ver README §7).

    Returns:
        Código de saída (0 = sucesso).
    """
    from src.conf import resolve_device, seed_everything

    log.info("Config composta:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)  # torch.device — auto: MPS ▸ CUDA ▸ CPU
    log.info(f"seed={cfg.seed} · device={device} · mode={cfg.mode}")

    mode = str(cfg.mode)
    if mode not in _DISPATCH:
        raise ValueError(
            f"mode inválido: '{mode}'. Use um de {sorted(_DISPATCH)} "
            f"(ex.: 'python main.py mode=train')."
        )

    try:
        return _DISPATCH[mode](cfg, device)
    except Exception as exc:  # noqa: BLE001
        log.exception(f"Falha no modo '{mode}': {exc}")
        raise SystemExit(1) from exc  # sai != 0 → o make para (sem falso "✓")


if __name__ == "__main__":
    main()
