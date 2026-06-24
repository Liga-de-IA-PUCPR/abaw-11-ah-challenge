"""Orquestração end-to-end do pipeline BAH (áudio + texto · síntese).

Expõe as funções de alto nível acionadas por ``main.py`` (dispatch por ``cfg.mode``):

- ``run_preprocess``: índice → extração de áudio (mp4→flac) → janelas → cache.
- ``run_featurize``: janelas → embedders (texto+áudio) + tabular → Parquet (1 linha/janela).
"""

from __future__ import annotations

from src.pipeline.featurize import run_featurize
from src.pipeline.preprocess import run_preprocess

__all__ = ["run_preprocess", "run_featurize"]
