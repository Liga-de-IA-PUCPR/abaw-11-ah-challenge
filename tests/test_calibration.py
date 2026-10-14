"""Testes da calibração do limiar (seleção ``bootstrap``) e da varredura val→test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.aggregation import (
    _macro_f1_grid,
    calibrate_threshold_from_video_scores,
)
from src.training.metrics import video_macro_f1

GRID = np.linspace(0.0, 1.0, 101)


def _noisy(n: int = 124, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.6).astype(np.int64)
    s = np.clip(0.35 + 0.3 * y + rng.normal(0, 0.2, n), 0, 1).astype(np.float32)
    return y, s


def test_vectorized_macro_f1_matches_sklearn():
    y, s = _noisy()
    expected = [video_macro_f1(y, (s >= t).astype(np.int64)) for t in GRID]
    assert np.allclose(_macro_f1_grid(y, s, GRID), expected)


def test_bootstrap_threshold_is_deterministic_and_reports_real_f1():
    y, s = _noisy()
    thr, f1 = calibrate_threshold_from_video_scores(s, y, selection="bootstrap")
    assert (thr, f1) == calibrate_threshold_from_video_scores(s, y, selection="bootstrap")
    assert 0.0 <= thr <= 1.0
    assert f1 == video_macro_f1(y, (s >= thr).astype(np.int64))


def test_bootstrap_lands_inside_the_gap_of_separable_scores():
    y = np.array([0] * 40 + [1] * 60)
    s = np.concatenate([np.linspace(0.05, 0.30, 40), np.linspace(0.70, 0.95, 60)])
    thr, f1 = calibrate_threshold_from_video_scores(s, y, selection="bootstrap")
    assert 0.30 < thr <= 0.70 and f1 == 1.0


def _write_preds(path: Path, y: np.ndarray, s: np.ndarray) -> None:
    path.mkdir(parents=True)
    ids = [f"v{i}" for i in range(len(y))]
    pd.DataFrame({"video_id": ids, "y_true": y, "y_proba": s}).to_csv(
        path / "predictions.csv", index=False
    )


def test_threshold_sweep_compares_every_strategy(tmp_path):
    run = tmp_path / "ensemble" / "exp0"
    _write_preds(run / "eval_val", *_noisy(seed=1))
    _write_preds(run / "eval_test", *_noisy(n=300, seed=2))
    script = Path(__file__).resolve().parents[1] / "scripts" / "threshold_sweep.py"
    out = subprocess.run(
        [sys.executable, str(script), str(run)], capture_output=True, text=True, check=True
    )
    saved = json.loads((run / "threshold_sweep.json").read_text())
    names = [r["strategy"] for r in saved["results"]]
    assert names == ["fixo_0.5", "argmax", "smooth", "base_rate", "bootstrap"]
    assert all(0.0 <= r["test_f1"] <= 1.0 for r in saved["results"])
    assert "exp0" in out.stdout and "bootstrap" in out.stdout
