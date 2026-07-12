"""Persistência, tracking (W&B + local), relatórios e submissão do pipeline BAH.

Síntese áudio + texto (sem vídeo). Cobre as DUAS famílias de modelo:

- ``random_forest`` (family="sklearn", 100% CPU): bundle joblib + ``WandbRun`` manual.
- ``cross_attention`` (family="lightning"): ``.ckpt`` do Lightning + sidecar JSON;
  o tracking W&B é feito pelo ``WandbLogger`` nativo cabeado no ``LightningTrainer`` (FASE 4).

Exports:
- WandbRun                         : wrapper tolerante a ausência (online|offline|disabled).
- save_rf_bundle / load_rf_bundle  : joblib p/ RandomForest.
- save_neural_sidecar / load_neural_sidecar : sidecar do checkpoint Lightning.
- resolve_latest_checkpoint        : localiza o artefato mais recente (joblib OU ckpt).
- Reporter                         : metrics.json + results.txt + plots (sempre roda).
- write_submission / validate_submission / predict_from_checkpoint.
"""

from __future__ import annotations

from src.outputs.checkpoint import (
    CheckpointBundle,
    NeuralSidecar,
    load_neural_sidecar,
    load_rf_bundle,
    resolve_latest_checkpoint,
    resolve_output_dir,
    save_neural_sidecar,
    save_rf_bundle,
)
from src.outputs.reporter import Reporter
from src.outputs.submission import (
    predict_from_checkpoint,
    validate_submission,
    write_submission,
)
from src.outputs.wandb_logger import WandbRun

__all__ = [
    "CheckpointBundle",
    "NeuralSidecar",
    "Reporter",
    "WandbRun",
    "load_neural_sidecar",
    "load_rf_bundle",
    "predict_from_checkpoint",
    "resolve_latest_checkpoint",
    "resolve_output_dir",
    "save_neural_sidecar",
    "save_rf_bundle",
    "validate_submission",
    "write_submission",
]
