"""Testes da calibração do limiar (seleção ``bootstrap``)."""

from __future__ import annotations

import numpy as np

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
