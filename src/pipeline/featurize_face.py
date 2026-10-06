"""Featurização de Face Landmarker (MediaPipe Tasks) sobre o Parquet existente."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl
from omegaconf import DictConfig

from src.data.windowing import load_window_index
from src.features.face_mesh import FaceMeshExtractor
from src.logger import get_logger
from src.pipeline.featurize_columns import merge_window_columns

log = get_logger("pipeline.featurize_face")


def run_featurize_face(cfg: DictConfig) -> dict[str, Any]:
    """Adiciona coluna ``face_landmarks`` ao Parquet de features (FASE 3+).

    Requer Parquet base (``mode=featurize``), o índice de janelas do ``mode=preprocess`` e
    os vídeos ``.mp4`` sob ``data.paths.data_root`` (+ grupo opcional ``vision``).
    Processa **por vídeo** (um ``VideoCapture`` por arquivo) p/ não reabrir o mp4
    a cada janela.
    """
    data = cfg.data
    force = bool(data.get("force_face", False))
    parquet_path = Path(data.paths.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet ausente: {parquet_path}. Rode 'mode=featurize' antes.")

    df = pl.read_parquet(parquet_path)
    if "face_landmarks" in df.columns and not force:
        log.info(f"Coluna face_landmarks já presente em {parquet_path} (use data.force_face=true).")
        return {"parquet_path": str(parquet_path), "cached": True}

    interim_dir = Path(data.paths.interim_dir)
    windows_index = interim_dir / "windows_index.parquet"
    windows = load_window_index(windows_index)
    log.info(f"{len(windows)} janelas para Face Landmarker.")

    face_cfg = cfg.get("face_embedder", {}) or {}
    extractor = FaceMeshExtractor(
        max_frames=int(face_cfg.get("max_frames", 2)),
        min_detection_confidence=float(face_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(face_cfg.get("min_tracking_confidence", 0.5)),
        refine_landmarks=bool(face_cfg.get("refine_landmarks", True)),
    )

    video_root = Path(data.paths.data_root)
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w.video_id].append(i)

    landmarks: list[list[float] | None] = [None] * len(windows)
    try:
        from tqdm import tqdm

        for _video_id, idxs in tqdm(
            by_video.items(), desc="face_mesh", unit="video", total=len(by_video)
        ):
            flat = extractor.flatten(extractor.extract([windows[i] for i in idxs], video_root))
            for i, vec in zip(idxs, flat, strict=True):
                landmarks[i] = vec.tolist()
    finally:
        extractor.close()

    # Join por janela + zeros p/ janelas sem landmarks + escrita atômica (mesmo helper das
    # colunas extras do plano MoE).
    merge_window_columns(parquet_path, windows, {"face_landmarks": landmarks})
    log.info(f"face_landmarks gravado em {parquet_path} (d_face={extractor.dim}).")

    return {
        "parquet_path": str(parquet_path),
        "d_face": extractor.dim,
        "n_windows": len(windows),
        "n_videos": len(by_video),
    }
