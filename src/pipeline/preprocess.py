"""Orquestração do pré-processamento BAH (FASE 2).

``run_preprocess(cfg)`` conecta as peças da FASE 2 em uma função idempotente:

    1. build_video_index(cfg)  → lista de ``VideoRecord`` (split/*.txt, *.yaml/.yml,
       transcription) — README §6.1.
    2. extract_audio           → para cada ``.mp4``, gera ``.flac`` 16 kHz mono
       (reusa ``src/scripts/extract_audio.py`` da branch ``matheus``: ffmpeg).
    3. WindowGenerator.generate → janela deslizante 5 s / hop 2.5 s, alinhando
       texto⟷áudio por timestamp do Whisper; rótulo da janela via *overlap* com
       ``time_detailed_ah``; rótulo do vídeo = ``global_ah``.
    4. cache do índice de janelas em ``data/interim/`` (Parquet) — uma linha por
       janela (sem embeddings ainda; estes vêm na FASE 3 / ``run_featurize``).

Respeita splits **participant-wise** (sem vazamento de participante). Reentrante:
usa cache de áudio; ``cfg.data.force=True`` recomputa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.logger import get_logger

log = get_logger("pipeline.preprocess")


def run_preprocess(cfg: DictConfig) -> dict[str, Any]:
    """Executa indexação → extração de áudio → janelamento → cache (FASE 2).

    Args:
        cfg: Config Hydra composta (usa ``cfg.data.*`` — README §7).

    Returns:
        Resumo com ``n_videos``, ``n_windows``, ``audio_dir`` e ``windows_index``
        (caminho do Parquet de janelas em ``data/interim/``).
    """
    # Imports tardios para manter o startup da CLI leve (sem torch no preprocess).
    from src.data.audio_io import extract_audio
    from src.data.indexing import build_video_index
    from src.data.windowing import WindowGenerator, save_window_index

    data = cfg.data
    force = bool(data.get("force", False))

    interim_dir = Path(data.paths.interim_dir)
    audio_dir = interim_dir / "Audio"  # data/interim/Audio/ (README §4)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Índice do dataset (todos os splits de uma vez) ---------------------
    log.info(f"Indexando o dataset em '{data.paths.data_root}'...")
    records = build_video_index(cfg)  # data_root default = data/raw/data
    log.info(f"{len(records)} vídeos indexados (train/val/test).")

    # --- 2. Extração de áudio (mp4 → flac 16 kHz mono, reusa extract_audio.py) --
    for rec in records:
        flac_path = audio_dir / rec.participant_id / f"{Path(rec.video_path).stem}.flac"
        if force or not flac_path.exists():
            extract_audio(
                mp4_path=rec.video_path,
                out_path=flac_path,
                sample_rate=data.audio.sample_rate,  # 16000
                mono=data.audio.mono,  # True
                overwrite=force,
            )
        rec.audio_path = flac_path

    # --- 3. Janelamento (deslizante + alinhamento texto⟷áudio + rótulo) --------
    window_gen = WindowGenerator(
        size_s=data.window.size_s,  # 5.0
        hop_s=data.window.hop_s,  # 2.5
        min_overlap_for_positive=data.window.min_overlap_for_positive,
        pad_last=data.window.pad_last,
    )
    windows: list = []
    for rec in records:
        windows.extend(window_gen.generate(rec))  # rótulo via time_detailed_ah; vídeo via global_ah
    log.info(
        f"{len(windows)} janelas geradas (size={data.window.size_s}s, hop={data.window.hop_s}s)."
    )

    # --- 4. Cache do índice de janelas (Parquet, 1 linha/janela; sem embeddings) -
    windows_index = interim_dir / "windows_index.parquet"
    save_window_index(windows, windows_index)
    log.info(f"Índice de janelas salvo em {windows_index}.")

    return {
        "n_videos": len(records),
        "n_windows": len(windows),
        "audio_dir": str(audio_dir),
        "windows_index": str(windows_index),
    }
