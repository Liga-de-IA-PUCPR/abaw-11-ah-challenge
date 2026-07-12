"""Featurização de Face Mesh (MediaPipe 468 pts) sobre o Parquet existente."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from omegaconf import DictConfig

from src.data.windowing import load_window_index
from src.features.face_mesh import FaceMeshExtractor
from src.logger import get_logger

log = get_logger("pipeline.featurize_face")


def run_featurize_face(cfg: DictConfig) -> dict[str, Any]:
    """Adiciona coluna ``face_landmarks`` ao Parquet de features (FASE 3+).

    Requer Parquet base (``mode=featurize``) e vídeos ``.mp4`` sob ``data.paths.data_root``.
    """
    data = cfg.data
    force = bool(data.get("force_face", False))
    parquet_path = Path(data.paths.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet ausente: {parquet_path}. Rode 'mode=featurize' antes."
        )

    df = pl.read_parquet(parquet_path)
    if "face_landmarks" in df.columns and not force:
        log.info(f"Coluna face_landmarks já presente em {parquet_path} (use data.force_face=true).")
        return {"parquet_path": str(parquet_path), "cached": True}

    interim_dir = Path(data.paths.interim_dir)
    windows_index = interim_dir / "windows_index.parquet"
    windows = load_window_index(windows_index)
    log.info(f"{len(windows)} janelas para Face Mesh.")

    face_cfg = cfg.get("face_embedder", {}) or {}
    extractor = FaceMeshExtractor(
        max_frames=int(face_cfg.get("max_frames", 8)),
        min_detection_confidence=float(face_cfg.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(face_cfg.get("min_tracking_confidence", 0.5)),
        refine_landmarks=bool(face_cfg.get("refine_landmarks", True)),
    )

    video_root = Path(data.paths.data_root)
    chunk_size = int(data.get("featurize_chunk_size", 64))
    rows: list[dict[str, Any]] = []

    try:
        from tqdm import tqdm

        chunk_starts = range(0, len(windows), chunk_size)
        for start in tqdm(chunk_starts, desc="face_mesh", unit="chunk"):
            batch = windows[start : start + chunk_size]
            lm = extractor.extract(batch, video_root)
            flat = extractor.flatten(lm)
            for w, vec in zip(batch, flat, strict=True):
                rows.append(
                    {
                        "id": w.video_id,
                        "t0": float(w.t0),
                        "t1": float(w.t1),
                        "face_landmarks": vec.tolist(),
                    }
                )
    finally:
        extractor.close()

    face_df = pl.DataFrame(rows)
    merged = df.join(face_df, on=["id", "t0", "t1"], how="left")
    if merged.filter(pl.col("face_landmarks").is_null()).height:
        n_miss = merged.filter(pl.col("face_landmarks").is_null()).height
        log.warning(f"{n_miss} janelas sem face_landmarks após join — preenchendo com zeros.")
        merged = merged.with_columns(
            pl.when(pl.col("face_landmarks").is_null())
            .then(pl.lit([0.0] * extractor.dim))
            .otherwise(pl.col("face_landmarks"))
            .alias("face_landmarks")
        )

    merged.write_parquet(parquet_path)
    log.info(f"face_landmarks gravado em {parquet_path} (d_face={extractor.dim}).")

    return {
        "parquet_path": str(parquet_path),
        "d_face": extractor.dim,
        "n_windows": len(windows),
    }
