"""Contratos de dados canônicos do pipeline BAH (ver README §6.1).

VideoRecord  — uma resposta de vídeo (áudio + transcrição + rótulos a nível de vídeo).
WindowSample — uma janela deslizante já alinhada (texto sobreposto + rótulo de janela).

Estas dataclasses são consumidas por TODAS as fases seguintes; mudanças aqui são
mudanças de contrato.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ==============================================================================
# Registro a nível de VÍDEO
# ==============================================================================


@dataclass
class VideoRecord:
    """Uma resposta de vídeo do BAH (áudio + texto; vídeo nunca é usado).

    Reúne tudo que o pipeline precisa para gerar janelas e rótulos:
    - identidade (video_id, participante, pergunta);
    - caminhos de mídia (mp4 + flac extraído);
    - transcrição Whisper (chunks com timestamp + texto completo);
    - rótulos: ``global_ah`` (nível de vídeo, ALVO FINAL) e ``time_detailed_ah``
      (intervalos com A/H, fonte do rótulo de cada janela);
    - metadados demográficos do participante (``meta_data.yml``).
    """

    video_id: str  # id usado em split/*.txt (caminho relativo do .mp4)
    participant_id: str
    question_id: int  # 1..7
    question_type: str  # "neutral"|"positive"|...|"hesitant"
    split: Literal["train", "val", "test"]
    video_path: Path  # .mp4 (apenas a faixa de áudio é usada)
    audio_path: Path | None  # .flac 16 kHz mono (preenchido após extração)
    duration_s: float
    transcript_chunks: list[dict]  # [{"start": float, "end": float, "text": str, "language": str}]
    full_transcript: str
    global_ah: int | None  # 0/1 — rótulo a nível de VÍDEO (alvo final); None no test
    time_detailed_ah: list[
        tuple[float, float]
    ]  # intervalos (s) com A/H — fonte do rótulo de janela
    certainty_ah: list[int]
    all_cues: list[dict]  # pistas dos anotadores (audio/body/facial/language/incons.)
    meta: dict  # metadados do participante (meta_data.yml)


# ==============================================================================
# Amostra a nível de JANELA
# ==============================================================================


@dataclass
class WindowSample:
    """Uma janela deslizante já alinhada áudio⟷texto.

    Produzida por :class:`~src.data.windowing.WindowGenerator`. O ``text`` já contém os
    chunks da transcrição sobrepostos a ``[t0, t1]``; o waveform correspondente é cortado
    sob demanda por :func:`~src.data.audio_io.load_segment` na FASE 3 (featurização).

    ``label`` é o rótulo da janela (0/1) derivado da sobreposição com ``time_detailed_ah``;
    é ``None`` no split de teste (sem rótulos). ``split`` e ``video_label`` viajam junto
    (herdados do ``VideoRecord``) para que o cache Parquet (FASE 3) carregue o split
    participant-wise e o alvo a nível de vídeo sem reconstruir o índice.
    """

    window_id: str  # f"{video_id}#w{idx}"
    video_id: str
    participant_id: str
    t0: float  # início (s)
    t1: float  # fim (s)
    text: str  # chunks de transcrição sobrepostos à janela (alinhado)
    label: int | None  # 0/1 via sobreposição com time_detailed_ah; None no test
    question_type: str
    split: Literal["train", "val", "test"]  # split do vídeo (participant-wise)
    video_label: int | None  # global_ah do vídeo (alvo de avaliação); None no test
    meta: dict = field(default_factory=dict)
