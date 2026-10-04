"""Fixtures compartilhadas: Parquet de janelas sintético (contrato do README §6.2 + colunas
extras do plano MoE) e a config Hydra real apontando para ele."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def write_window_parquet(path: Path, n_participants: int = 24, seed: int = 0) -> Path:
    """2 vídeos por participante, 3–6 janelas cada; o rótulo do vídeo vaza fraco nas features
    (os modelos têm o que aprender). Splits por participante: 2/3 train, 1/6 val, 1/6 test."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {k: [] for k in (
        "id", "window_idx", "t0", "t1", "participant_id", "question_type", "audio_emb",
        "text_emb", "tabular", "label", "video_label", "split", "transcript", "asr_timing",
        "hesitation_markers", "audio_emb_x",
    )}  # fmt: skip
    w = 0
    for p in range(n_participants):
        split = "train" if p % 6 < 4 else ("val" if p % 6 == 4 else "test")
        for q in range(2):
            y = int(rng.random() < 0.5)
            asr = (rng.normal(y, 1.0, 5)).tolist()
            markers = (rng.normal(y, 1.0, 3)).tolist()
            text = "i think maybe not " * (1 + y) + "but it is fine"
            for t in range(int(rng.integers(3, 7))):
                rows["id"].append(f"Videos/{p}/v{q}.mp4")
                rows["window_idx"].append(w)
                rows["t0"].append(2.5 * t)
                rows["t1"].append(2.5 * t + 5.0)
                rows["participant_id"].append(str(p))
                rows["question_type"].append("neutral")
                rows["audio_emb"].append(rng.normal(y, 1.0, 4).tolist())
                rows["text_emb"].append(rng.normal(y, 1.0, 6).tolist())
                rows["tabular"].append(rng.normal(0, 1.0, 3).tolist())
                rows["label"].append(-1 if split == "test" else int(y and t % 2 == 0))
                rows["video_label"].append(y)
                rows["split"].append(split)
                rows["transcript"].append(text)
                rows["asr_timing"].append(asr)
                rows["hesitation_markers"].append(markers)
                rows["audio_emb_x"].append(rng.normal(y, 1.0, 4).tolist())
                w += 1
    pl.DataFrame(rows).with_columns(
        pl.col("window_idx").cast(pl.Int32),
        pl.col("t0").cast(pl.Float32),
        pl.col("t1").cast(pl.Float32),
        pl.col("label").cast(pl.Int8),
        pl.col("video_label").cast(pl.Int8),
    ).write_parquet(path)
    return path


@pytest.fixture
def window_parquet(tmp_path) -> Path:
    return write_window_parquet(tmp_path / "windows.parquet")


@pytest.fixture
def compose_cfg(tmp_path, window_parquet):
    """``compose_cfg(*overrides)`` → config Hydra real com dados/saídas em ``tmp_path``."""
    from hydra import compose, initialize_config_dir

    def _compose(*overrides: str):
        with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
            return compose(
                config_name="config",
                overrides=[
                    f"data.paths.parquet_path={window_parquet}",
                    f"data.paths.output_root={tmp_path / 'outputs'}",
                    "wandb.mode=disabled",
                    "device=cpu",
                    *overrides,
                ],
            )

    return _compose
