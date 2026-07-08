"""Persistência, relatórios e submissão do pipeline BAH.

Síntese áudio + texto (sem vídeo). Cobre as DUAS famílias de modelo:

- ``sklearn`` (100% CPU): ``model.joblib`` + ``trainer_state.json`` (``SklearnTrainer.save``).
- ``lightning`` (``cross_attention``): ``.ckpt`` do Lightning; o tracking W&B é feito
  pelo ``WandbLogger`` nativo cabeado no ``LightningTrainer`` (FASE 4).

Exports:
- resolve_output_dir / resolve_latest_checkpoint : diretórios de run (novo / mais recente).
- Reporter                                       : metrics.json + results.txt + plots.
- write_submission / validate_submission         : arquivo de submissão (FASE 5).
"""

from __future__ import annotations

from src.outputs.checkpoint import (
    resolve_latest_checkpoint,
    resolve_output_dir,
)
from src.outputs.reporter import Reporter
from src.outputs.submission import (
    validate_submission,
    write_submission,
)

__all__ = [
    "Reporter",
    "resolve_latest_checkpoint",
    "resolve_output_dir",
    "validate_submission",
    "write_submission",
]
