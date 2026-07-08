"""Pipeline de dados do BAH (áudio + texto).

Fornece:
- VideoRecord / WindowSample: contratos canônicos (ver README §6.1)
- build_video_index: parse de split/*.txt + *.yaml + *.yml + transcription → list[VideoRecord]
- extract_audio / load_segment: mp4 → flac 16 kHz mono (reusa scripts/extract_audio.py) + corte
- WindowGenerator: janela deslizante + alinhamento áudio⟷texto + rótulo de janela
- WindowMatrixView: matriz achatada por janela (random_forest, sklearn/CPU)
- VideoSequenceDataset / collate_sequences: sequência de janelas por vídeo (cross_attention, T>1)
"""

from __future__ import annotations

from src.data.audio_io import extract_audio, load_segment, pad_or_trim
from src.data.datasets import (
    VideoSequenceDataset,
    WindowMatrixView,
    collate_sequences,
    load_split,
    load_train_val,
)
from src.data.indexing import (
    QUESTION_TYPE_MAP,
    build_video_index,
    parse_video_filename,
)
from src.data.schema import VideoRecord, WindowSample
from src.data.windowing import (
    WindowGenerator,
    load_window_index,
    overlap_seconds,
    save_window_index,
)

__all__ = [
    "VideoRecord",
    "WindowSample",
    "build_video_index",
    "QUESTION_TYPE_MAP",
    "parse_video_filename",
    "extract_audio",
    "load_segment",
    "pad_or_trim",
    "WindowGenerator",
    "overlap_seconds",
    "save_window_index",
    "load_window_index",
    "WindowMatrixView",
    "VideoSequenceDataset",
    "collate_sequences",
    "load_train_val",
    "load_split",
]
