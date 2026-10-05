"""Entrypoint Hydra do desafio BAH (áudio + texto · síntese).

`@hydra.main` compõe a config a partir dos grupos (README §7) e faz *dispatch*
por ``cfg.mode``:

    preprocess     índice (FASE 2) + extração de áudio (mp4→flac 16 kHz) + janelas → cache
    featurize      janelas → embedders (texto+áudio) + tabular → Parquet (FASE 3)
    featurize_face (opcional, vídeo) Face Mesh por janela → coluna face_landmarks no Parquet
    featurize_columns  colunas extras do plano MoE no Parquet (transcrição, ASR timing,
                   marcadores de hesitação, áudio/rosto/cena por embedder) — ``columns=[...]``
    oof            protocolo OOF (5 dobras por participante, τ fixo) + gate pareado
                   (``oof.baseline``) — a régua de cada rodada do plano MoE
    route          MoERouter sobre runs OOF (``route.members``): combina os membros por
                   amostra a partir dos vetores pré-logit; gate vs melhor membro e média
    train          treina o modelo escolhido em ``model=`` (FASE 4)
    evaluate       avalia a nível de vídeo (Macro-F1, AP) em um split (FASE 4/5)
    submit         escreve o arquivo de submissão (video_id, pred) (FASE 5)
    hard_mining    (lightning) pesos de amostragem {video_id: peso} a partir de um checkpoint

Reusa a estrutura do ``main.py`` da branch ``matheus`` (Hydra + WandbLogger +
L.Trainer), **generalizada para todos os modelos do registry**: o modelo e o
trainer vêm das *factories* da FASE 4 (``create_model`` + ``create_trainer``),
de modo que o caminho ``random_forest`` **nunca importa Lightning**.

Modelos (``model=`` / presets ``+experiment=``): ``random_forest`` (sklearn),
``cross_attention`` (áudio+texto, artigo) e os GNNs do Rodrigo — ``hetero_gnn_contrastive``
(áudio+texto+tabular em grafo; preset ``hetero_gnn_v2_tune_wav2vec2``), ``face_gnn_ts`` e
``multimodal_hetero_face`` (+ vídeo; presets ``face_gnn_ts_roi``, ``multimodal_hetero_face_v2``). ``ensemble=[...]`` combina run dirs de QUALQUER
um deles (evaluate/submit).

Exemplos::

    # baseline RandomForest (CPU, sem Lightning) — usa todos os defaults
    python main.py

    # pré-processamento e featurização (embeddings em Apple Metal)
    python main.py mode=preprocess
    python main.py mode=featurize device=mps

    # modelo de competição (preset cross_attention) em MPS
    PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py +experiment=cross_attention device=mps

    # GNN heterogêneo do Rodrigo com o pré-processamento da main (cache wav2vec2)
    python main.py +experiment=hetero_gnn_v2_tune_wav2vec2 device=cuda

    # vídeo: landmarks no Parquet → GNN facial
    python main.py mode=featurize_face +face_embedder=mediapipe
    python main.py +experiment=face_gnn_ts_roi

    # ensemble heterogêneo: seeds CA (cache librosa) + GNN que lê o próprio cache wav2vec2
    python main.py mode=evaluate split=test +experiment=cross_attention \
        "ensemble=[outputs/cross_attention/A,outputs/cross_attention/B,\
    {checkpoint:outputs/hetero_gnn_contrastive/C,parquet_path:data/processed/text_audio_windows_w2v.parquet}]"
    # (atalho: make ensemble-multimodal GNN_RUN=... FACE_RUN=...)

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
            f"Featurize pulado (cache): {summary['n_windows']} janelas → "
            f"{summary['parquet_path']}. Use 'data.force=true' p/ recomputar."
        )
    else:
        log.info(
            f"Featurize concluído: {summary['n_windows']} janelas → {summary['parquet_path']} "
            f"(d_text={summary['d_text']}, d_audio={summary['d_audio']}, d_tab={summary['d_tab']})."
        )
    return 0


def _as_loader(cfg: DictConfig, data, family: str, split: str):
    """Adapta ``data`` ao trainer da ``family`` (ver :func:`src.data.datasets.as_loader`)."""
    from src.data.datasets import as_loader

    return as_loader(cfg, data, family, split)


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
    # O split de calibração pode não ser 'val' (ex.: 'test' na metodologia externa) — o log
    # reflete qual foi, para não enganar (as métricas de result['val'] são desse split).
    calib_name = cfg.data.get("calib_split", "val")
    log.info(
        f"Treino concluído (family={family}). "
        f"Macro-F1({calib_name})={result['val']['macro_f1']:.4f} | "
        f"AP({calib_name})={result['val'].get('average_precision', 0.0):.4f} | "
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


def _maybe_recalibrate(cfg: DictConfig, trainer, family: str) -> None:
    """Recalibra o limiar na val (sem re-treinar) se pedido.

    O limiar salvo no checkpoint é congelado; mudar a estratégia de calibração só
    valeria num re-treino. Com ``aggregation.recalibrate=true``, evaluate/submit rodam
    inferência na val e recalculam ``threshold_`` com a config atual. **Ensemble sempre
    recalibra** (o limiar médio dos membros não vale para a proba média). Só age se
    ``threshold=="auto"`` (um float fixo explícito vence).
    """
    agg = getattr(cfg, "aggregation", None)
    if agg is None:
        return
    if agg.get("threshold", "auto") not in (None, "auto"):
        return  # limiar fixo explícito tem prioridade sobre recalibrar
    want = bool(agg.get("recalibrate", False)) or bool(cfg.get("ensemble"))
    if not want:
        return
    if not hasattr(trainer, "recalibrate_on_val"):
        log.warning(f"family={family} não suporta recalibrar-na-val; mantendo limiar salvo.")
        return
    from src.data.datasets import load_split

    # Recalibra no MESMO split usado no treino (data.calib_split), lido de
    # data.paths.calib_parquet_path se definido (senão do parquet de predição). Isso
    # permite PREDIZER num parquet (ex.: externo) e CALIBRAR noutro (ex.: raw test).
    calib_split = cfg.data.get("calib_split", "val")
    calib_pq = cfg.data.paths.get("calib_parquet_path", None)
    data_cfg = _data_cfg(cfg, trainer)
    calib_view = load_split(data_cfg, calib_split, family=family, parquet_path=calib_pq)
    calib_loader = _as_loader(cfg, calib_view, family, str(calib_split))
    trainer.recalibrate_on_val(calib_loader)


def _data_cfg(cfg: DictConfig, trainer) -> DictConfig:
    """Config que descreve os dados do trainer carregado.

    Um run recarregado recria o modelo pelo ``model_config`` do treino (``load_trainer``),
    que pode diferir do ``model`` da CLI (ex.: ramos extras do ``moe_fusion``) — os dados
    (colunas/transcrição pedidas pelo modelo) seguem a config do próprio trainer.
    """
    return getattr(trainer, "config", None) or cfg


def _member_spec(item) -> dict:
    """Normaliza um membro de ``ensemble``: run dir (str) ou dict com ``checkpoint``."""
    if isinstance(item, str):
        return {"checkpoint": item}
    spec = OmegaConf.to_container(item, resolve=True) if not isinstance(item, dict) else dict(item)
    if not spec.get("checkpoint"):
        raise ValueError(f"Membro de ensemble sem 'checkpoint': {spec}")
    return spec


def _cfg_for_model(cfg: DictConfig, model_name: str, experiment: str | None = None) -> DictConfig:
    """``cfg`` do membro: o do run, com o grupo ``model`` do modelo do membro.

    - ``experiment`` dado (runs antigos treinados por preset): o ``model`` vem do preset
      composto (``+experiment=<experiment>``) — mesma arquitetura do treino.
    - Mesmo modelo do run: ``cfg`` intacto (reprodutibilidade do ensemble homogêneo).
    - Outro modelo: ``configs/model/<model_name>.yaml``.
    Em todos os casos, se o run gravou ``model_config`` (runs novos), o ``load_trainer``
    recria o modelo a partir dele — o que é escolhido aqui só vale para runs antigos.
    """
    from pathlib import Path

    if experiment:
        from hydra import compose

        composed = compose(config_name="config", overrides=[f"+experiment={experiment}"])
        member_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        member_cfg.model = composed.model
        return member_cfg
    if model_name == str(cfg.model.name):
        return cfg
    model_yaml = Path(__file__).resolve().parent / "configs" / "model" / f"{model_name}.yaml"
    if not model_yaml.exists():
        raise FileNotFoundError(f"Membro '{model_name}': YAML ausente em {model_yaml}")
    member_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    member_cfg.model = OmegaConf.load(model_yaml)
    return member_cfg


def _load_ensemble(cfg: DictConfig):
    """Monta o :class:`EnsembleTrainer` a partir de ``cfg.ensemble`` (membros heterogêneos).

    Cada membro é um run dir de QUALQUER modelo do registry. O nome do modelo vem, nesta
    ordem, de ``model`` no dict do membro → ``model_name`` do ``trainer_state.json`` →
    pasta-pai do run (``outputs/<modelo>/<timestamp>``). ``parquet_path`` no dict faz o
    membro ler as SUAS features (ex.: GNN no cache wav2vec2) para os mesmos vídeos;
    ``experiment`` recompõe o ``model`` do preset do treino (runs antigos, sem
    ``model_config`` no ``trainer_state.json``).
    Retorna ``(trainer, report_dir)``.
    """
    import json
    from pathlib import Path

    import src.models  # noqa: F401  — registra os modelos (famílias p/ get_family)
    from src.models.registry import get_family
    from src.training.ensemble import EnsembleMember, EnsembleTrainer
    from src.training.factory import load_trainer

    specs = [_member_spec(x) for x in cfg.ensemble]
    weights = cfg.get("ensemble_weights")
    if weights is not None and len(weights) != len(specs):
        raise ValueError(f"ensemble_weights tem {len(weights)} pesos p/ {len(specs)} membros.")

    members: list[EnsembleMember] = []
    for i, spec in enumerate(specs):
        ckpt = str(spec["checkpoint"])
        state_file = Path(ckpt) / "trainer_state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        model_name = str(spec.get("model") or state.get("model_name") or Path(ckpt).parent.name)
        family = get_family(model_name)
        member_cfg = _cfg_for_model(cfg, model_name, spec.get("experiment"))
        trainer = load_trainer(family, ckpt, cfg=member_cfg)
        weight = spec.get("weight", weights[i] if weights is not None else 1.0)
        members.append(
            EnsembleMember(
                trainer=trainer,
                name=ckpt,
                family=family,
                parquet_path=spec.get("parquet_path"),
                calib_parquet_path=spec.get("calib_parquet_path"),
                weight=float(weight),
            )
        )

    # Relatórios: outputs/<modelo>/<nome> se todos os membros são do mesmo modelo (layout
    # original, ex.: cross_attention/ensemble_5); senão outputs/ensemble/<nome>.
    names = {Path(m.name).parent.name for m in members}
    group = names.pop() if len(names) == 1 else "ensemble"
    # +ensemble_name=<nome> isola os relatórios de ensembles distintos com o mesmo nº
    # de membros (ex.: ensemble_5 vs ensemble_5_all); default mantém ensemble_<N>.
    ens_name = cfg.get("ensemble_name") or f"ensemble_{len(members)}"
    report_dir = Path(cfg.data.paths.output_root) / group / str(ens_name)
    # Loga QUAIS checkpoints entram (auditabilidade: confirma o ensemble usado) +
    # o limiar individual de cada membro (calibrado no treino).
    member_info = "\n".join(
        f"    [{i}] {m.name}  (family={m.family}, thr_membro={m.trainer.threshold_}"
        + (f", peso={m.weight:g}" if m.weight != 1.0 else "")
        + (f", parquet={m.parquet_path}" if m.parquet_path else "")
        + ")"
        for i, m in enumerate(members)
    )
    log.info(f"Ensemble de {len(members)} checkpoints → relatórios em {report_dir}:\n{member_info}")
    return EnsembleTrainer(members, cfg), str(report_dir)


def _resolve_trainer(cfg: DictConfig, family: str):
    """Carrega UM checkpoint ou um ENSEMBLE (média de probas). Retorna ``(trainer, report_dir)``.

    ``ensemble=[...]`` monta um :class:`EnsembleTrainer` que média as probas por vídeo dos
    membros (ver :func:`_load_ensemble`); senão resolve 1 checkpoint (``cfg.checkpoint`` ou
    o mais recente da família). O ``report_dir`` é onde ``_write_eval_report`` grava.
    """
    from src.outputs.checkpoint import resolve_latest_checkpoint
    from src.training.factory import load_trainer

    if cfg.get("ensemble"):
        return _load_ensemble(cfg)

    # Sem checkpoint explícito: o run mais recente DO MODELO atual (outputs/<model.name>/…) —
    # com vários modelos lightning convivendo (CA + GNNs), filtrar só pela família poderia
    # carregar o run de outro modelo.
    ckpt_dir = cfg.get("checkpoint") or resolve_latest_checkpoint(
        cfg.data.paths.output_root, family=family, model_name=str(cfg.model.name)
    )
    return load_trainer(family, ckpt_dir, cfg=cfg), ckpt_dir


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

    # Plots a nível de vídeo (confusão, ROC, PR, curva de limiar) + predictions.csv por
    # vídeo (insumo do meta-router CA⊕GNN e do protocolo OOF) — exige video_outputs.
    if hasattr(trainer, "video_outputs"):
        o = None
        try:
            o = trainer.video_outputs(data)
            rep.plot_confusion_matrix(o["y_true"], o["y_pred"])
            rep.plot_roc_curve(o["y_true"], o["y_proba"])
            rep.plot_precision_recall(o["y_true"], o["y_proba"])
            grid, f1s = threshold_curve(o["y_true"], o["y_proba"])
            rep.plot_threshold_curve(grid, f1s, threshold)
        except Exception as exc:  # noqa: BLE001 — plots nunca derrubam o evaluate
            log.warning(f"Plots a nível de vídeo pulados: {exc}")
        if o is not None:
            try:
                meta = load_video_metadata_for_eval(cfg, o["video_ids"])
                rep.save_predictions_csv(
                    o["video_ids"], o["y_true"], o["y_proba"], o["y_pred"], threshold, metadata=meta
                )
                if "embedding" in o:  # vetor pré-logit (moe_fusion) → MoERouter
                    rep.save_embeddings(o["embedding"])
                rep.save_error_analysis(o["video_ids"], o["y_true"], o["y_pred"], metadata=meta)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"predictions.csv / análise de erro pulados: {exc}")

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

    family = _data_family(cfg, device)
    # Resolve 1 checkpoint (mais recente da família) OU um ensemble (ensemble=[...]).
    trainer, ckpt_dir = _resolve_trainer(cfg, family)
    _apply_threshold_override(cfg, trainer)  # aggregation.threshold=<float> sobrepõe o salvo
    _maybe_recalibrate(cfg, trainer, family)  # recalibrate=true OU ensemble → recalibra na val
    log.info(f"Checkpoint carregado: {ckpt_dir}")

    split = cfg.get("split") or "val"
    view = load_split(_data_cfg(cfg, trainer), split, family=family)
    data = _as_loader(cfg, view, family, split)
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
    from src.outputs.submission import read_reference_order, write_submission

    family = _data_family(cfg, device)
    # Resolve 1 checkpoint OU um ensemble (ensemble=[...]) — média de probas por vídeo.
    trainer, _ = _resolve_trainer(cfg, family)
    _apply_threshold_override(cfg, trainer)  # aggregation.threshold=<float> sobrepõe o salvo
    _maybe_recalibrate(cfg, trainer, family)  # recalibrate=true OU ensemble → recalibra na val

    split = cfg.get("split") or "test"
    out_path = Path(cfg.get("out") or "outputs/submission.txt")
    view = load_split(_data_cfg(cfg, trainer), split, family=family)
    data = _as_loader(cfg, view, family, split)

    # Formato oficial do desafio (README §9): ordem da referência + (opcional) probabilidades.
    # submission_reference = caminho do trial-0.txt de referência (define a ORDEM exigida).
    # submission_probabilities = escreve 'video_id,p0,p1,pred' (habilita o AP) em vez de 'video_id,pred'.
    order = read_reference_order(cfg.get("submission_reference"))

    want_probs = bool(cfg.get("submission_probabilities", False))
    if hasattr(trainer, "predict_scores"):
        # Uma inferência só: deriva os labels dos scores (evita rodar o ensemble 2x).
        scores = trainer.predict_scores(data)  # {video_id: p1}
        thr = float(trainer.threshold_)
        video_preds = {v: int(p >= thr) for v, p in scores.items()}
        probs = scores if want_probs else None
    else:  # trainer sem score contínuo por vídeo exposto → só labels
        if want_probs:
            log.warning(f"family={family} não expõe predict_scores; submissão sem probabilidades.")
        video_preds = trainer.predict(data)
        probs = None

    write_submission(video_preds, out_path, order=order, probabilities=probs)
    log.info(f"Submissão escrita: {out_path} ({len(video_preds)} vídeos).")
    return 0


def _data_family(cfg: DictConfig, device) -> str:
    """Família da VISÃO de dados do evaluate/submit.

    Ensemble → sempre ``lightning`` (sequência por vídeo; membros sklearn montam a própria
    matriz de janelas dentro do ``EnsembleTrainer``). Senão, a família do ``model=``.
    """
    if cfg.get("ensemble"):
        return "lightning"
    _, family = _build_trainer(cfg, device)
    return family


def _run_featurize_face(cfg: DictConfig) -> int:
    """``mode=featurize_face`` — MediaPipe Face Mesh → coluna ``face_landmarks`` no Parquet.

    Requer o Parquet base (``mode=featurize``), os ``.mp4`` sob ``data.paths.data_root`` e o
    grupo opcional ``vision`` (``uv sync --group vision``).
    """
    from src.pipeline.featurize_face import run_featurize_face

    summary = run_featurize_face(cfg)
    if summary.get("cached"):
        log.info(
            f"Featurize face pulado (cache): {summary['parquet_path']} "
            "(use data.force_face=true p/ recomputar)."
        )
    else:
        log.info(
            f"Featurize face concluído: {summary.get('n_windows', '?')} janelas → "
            f"{summary['parquet_path']} (d_face={summary.get('d_face', '?')})."
        )
    return 0


def _run_featurize_columns(cfg: DictConfig) -> int:
    """``mode=featurize_columns`` — grava ``columns=[...]`` como colunas extras do Parquet.

    Requer o Parquet base (``mode=featurize``) e o índice de janelas (``mode=preprocess``);
    colunas de vídeo/rosto leem o dataset bruto (``data.paths.data_root``).
    """
    from src.pipeline.featurize_columns import run_featurize_columns

    summary = run_featurize_columns(cfg)
    if summary.get("cached"):
        log.info(f"Colunas já presentes em {summary['parquet_path']} (data.force_columns=true).")
    else:
        log.info(f"Colunas {summary['columns']} gravadas em {summary['parquet_path']}.")
    return 0


def _run_oof(cfg: DictConfig) -> int:
    """``mode=oof`` — predições out-of-fold + métricas com τ fixo + gate vs ``oof.baseline``."""
    import src.models  # noqa: F401  — registra os modelos
    from src.pipeline.oof import run_oof

    summary = run_oof(cfg)
    log.info(f"OOF concluído: {summary['out_dir']}")
    return 0


def _run_route(cfg: DictConfig) -> int:
    """``mode=route`` — MoERouter (avaliado nas dobras OOF dos membros) + predição final."""
    from src.pipeline.route import run_route

    summary = run_route(cfg)
    log.info(f"MoERouter concluído: {summary['out_dir']}")
    return 0


def _run_hard_mining(cfg: DictConfig, device) -> int:
    """``mode=hard_mining`` — pontua um split (default train) e grava pesos de amostragem.

    Roda o checkpoint sobre o split, marca *hard positives* (rótulo 1, proba baixa),
    *hard negatives* (rótulo 0, proba alta), casos limítrofes (perto do limiar) e tipos
    de pergunta raros, e grava ``{video_id: peso}`` em JSON. O treino consome via
    ``data.hard_examples=<json>`` (``WeightedRandomSampler``). Parâmetros opcionais por
    CLI: ``+hard_hi``, ``+hard_lo``, ``+hard_weight``, ``+hard_border``,
    ``+hard_border_weight``, ``+hard_rare_types``, ``+hard_rare_mult``.
    """
    import json
    from pathlib import Path

    from src.data.datasets import load_split
    from src.outputs.reporter import load_video_metadata_for_eval

    family = _data_family(cfg, device)
    if family != "lightning":
        raise ValueError("mode=hard_mining requer um modelo da família lightning.")
    trainer, ckpt_dir = _resolve_trainer(cfg, family)
    log.info(f"Hard mining: checkpoint {ckpt_dir}")

    split = cfg.get("split") or "train"
    data = _as_loader(
        cfg, load_split(_data_cfg(cfg, trainer), split, family=family), family, "eval"
    )
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
        yt, p = int(o["y_true"][i]), float(o["y_proba"][i])
        w = 1.0
        if (yt == 1 and p < hard_lo) or (yt == 0 and p > hard_hi):
            w *= hard_weight
            n_hard += 1
        elif abs(p - threshold) < border_margin:
            w *= border_weight
            n_border += 1
        if (meta.get(vid, {}) or {}).get("question_type") in rare_types:
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


_DISPATCH = {
    "preprocess": lambda cfg, dev: _run_preprocess(cfg),
    "featurize": _run_featurize,
    "featurize_face": lambda cfg, dev: _run_featurize_face(cfg),
    "featurize_columns": lambda cfg, dev: _run_featurize_columns(cfg),
    "oof": lambda cfg, dev: _run_oof(cfg),
    "route": lambda cfg, dev: _run_route(cfg),
    "train": _run_train,
    "evaluate": _run_evaluate,
    "submit": _run_submit,
    "hard_mining": _run_hard_mining,
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
