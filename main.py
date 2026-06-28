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

    return DataLoader(
        data,
        batch_size=cfg.data.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.data.num_workers,
        collate_fn=collate_sequences,
    )


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

    result = trainer.fit(train_data, val_data)
    log.info(
        f"Treino concluído (family={family}). "
        f"Macro-F1(val)={result['val']['macro_f1']:.4f} | "
        f"AP(val)={result['val'].get('average_precision', 0.0):.4f} | "
        f"limiar={result.get('threshold', 0.5):.4f}."
    )

    out_dir = resolve_output_dir(cfg.data.paths.output_root, cfg.model.name)
    trainer.save(out_dir)
    log.info(f"Checkpoint salvo em: {out_dir}")
    return 0


def _run_evaluate(cfg: DictConfig, device) -> int:
    """``mode=evaluate`` — métricas a nível de vídeo num split (FASE 4/5)."""
    import json

    from src.data.datasets import load_split
    from src.outputs.checkpoint import resolve_latest_checkpoint
    from src.training.factory import load_trainer

    _, family = _build_trainer(cfg, device)
    # resolve_latest_checkpoint(output_root) — UM argumento (FASE 5); varre as duas
    # famílias (bundle.joblib | *.ckpt) sob a raiz e devolve o mais recente.
    ckpt_dir = cfg.get("checkpoint") or resolve_latest_checkpoint(cfg.data.paths.output_root)
    trainer = load_trainer(family, ckpt_dir, cfg=cfg)
    log.info(f"Checkpoint carregado: {ckpt_dir}")

    split = cfg.get("split") or "val"
    data = _as_loader(cfg, load_split(cfg, split, family=family), family, split)
    report = trainer.evaluate(data)
    log.info(
        f"[{split}] Macro-F1={report['macro_f1']:.4f} | "
        f"AP={report.get('average_precision', 0.0):.4f} (n={report['n_videos']})."
    )
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
    # resolve_latest_checkpoint(output_root) — UM argumento (FASE 5).
    ckpt_dir = cfg.get("checkpoint") or resolve_latest_checkpoint(cfg.data.paths.output_root)
    trainer = load_trainer(family, ckpt_dir, cfg=cfg)

    split = cfg.get("split") or "test"
    out_path = Path(cfg.get("out") or "outputs/submission.txt")
    data = _as_loader(cfg, load_split(cfg, split, family=family), family, split)
    video_preds = trainer.predict(data)  # {video_id: 0/1}
    write_submission(video_preds, out_path)
    log.info(f"Submissão escrita: {out_path} ({len(video_preds)} vídeos).")
    return 0


_DISPATCH = {
    "preprocess": lambda cfg, dev: _run_preprocess(cfg),
    "featurize": _run_featurize,
    "train": _run_train,
    "evaluate": _run_evaluate,
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
