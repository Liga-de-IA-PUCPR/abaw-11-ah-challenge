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

log = get_logger("pipeline.featurize_face")


def run_featurize_face(cfg: DictConfig) -> dict[str, Any]:
    """Adiciona coluna ``face_landmarks`` ao Parquet de features (FASE 3+).

    Requer Parquet base (``mode=featurize``), o índice de janelas do ``mode=preprocess`` e
    os vídeos ``.mp4`` sob ``data.paths.data_root`` (+ grupo opcional ``vision``).
    Processa **por vídeo** (um ``VideoCapture`` por arquivo) p/ não reabrir o mp4
    a cada janela.

    Com ``data.face_from=<parquet>`` não roda o MediaPipe: copia a coluna ``face_landmarks``
    daquele Parquet (mesmas janelas, outro cache de áudio/texto) — os landmarks são
    extraídos uma vez e reaproveitados por todos os caches.
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

    face_from = data.get("face_from")
    if face_from:
        return _copy_face_column(df, parquet_path, Path(face_from))

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
    by_video: dict[str, list] = defaultdict(list)
    for w in windows:
        by_video[w.video_id].append(w)

    rows: list[dict[str, Any]] = []
    try:
        from tqdm import tqdm

        for _video_id, batch in tqdm(
            by_video.items(), desc="face_mesh", unit="video", total=len(by_video)
        ):
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

    _write_with_face(df, pl.DataFrame(rows), parquet_path, extractor.dim)
    return {
        "parquet_path": str(parquet_path),
        "d_face": extractor.dim,
        "n_windows": len(windows),
        "n_videos": len(by_video),
    }


def _copy_face_column(df: pl.DataFrame, parquet_path: Path, source: Path) -> dict[str, Any]:
    """Grava em ``parquet_path`` a coluna ``face_landmarks`` lida de ``source`` (sem MediaPipe)."""
    if not source.exists():
        raise FileNotFoundError(f"data.face_from ausente: {source}")
    face_df = pl.read_parquet(source, columns=["id", "t0", "t1", "face_landmarks"])
    d_face = len(face_df["face_landmarks"][0])
    log.info(f"Copiando face_landmarks de {source} → {parquet_path} (sem MediaPipe).")
    _write_with_face(df.drop("face_landmarks", strict=False), face_df, parquet_path, d_face)
    return {
        "parquet_path": str(parquet_path),
        "d_face": d_face,
        "n_windows": df.height,
        "n_videos": df["id"].n_unique(),
        "face_from": str(source),
    }


def _write_with_face(
    df: pl.DataFrame, face_df: pl.DataFrame, parquet_path: Path, d_face: int
) -> None:
    """Join por janela ``(id, t0, t1)`` + zeros onde não há rosto + escrita atômica."""
    merged = df.join(face_df, on=["id", "t0", "t1"], how="left")
    n_miss = merged.filter(pl.col("face_landmarks").is_null()).height
    if n_miss:
        log.warning(f"{n_miss} janelas sem face_landmarks após join — preenchendo com zeros.")
        merged = merged.with_columns(
            pl.when(pl.col("face_landmarks").is_null())
            .then(pl.lit([0.0] * d_face))
            .otherwise(pl.col("face_landmarks"))
            .alias("face_landmarks")
        )

    # Escrita atômica: o Parquet base (caro de recomputar) só é substituído quando o novo
    # arquivo está completo — um crash no meio não corrompe o cache.
    tmp_path = parquet_path.with_suffix(".face_tmp.parquet")
    merged.write_parquet(tmp_path)
    tmp_path.replace(parquet_path)
    log.info(f"face_landmarks gravado em {parquet_path} (d_face={d_face}).")
