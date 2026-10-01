"""Testes de `src/eval/protocol.py` — CV agrupada por participante + OOF."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score

from src.eval.protocol import (
    average_precision_score_,
    bootstrap_ci_macro_f1,
    macro_f1_at_threshold,
    make_group_folds,
    paired_bootstrap_macro_f1,
    run_oof,
)


def _toy_df(
    n_participants: int = 40, videos_per_participant: int = 3, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    vid = 0
    for p in range(n_participants):
        n_videos = rng.integers(1, videos_per_participant + 1)
        # Rótulo tende a ser consistente dentro do participante, com ruído.
        base_label = int(rng.integers(0, 2))
        for _ in range(n_videos):
            label = base_label if rng.random() > 0.2 else 1 - base_label
            rows.append(
                {
                    "video_id": f"v{vid}",
                    "participant_id": f"p{p}",
                    "label": label,
                    "x": rng.normal(loc=float(label), scale=1.0),
                }
            )
            vid += 1
    return pd.DataFrame(rows)


# ==============================================================================
# make_group_folds
# ==============================================================================


def test_no_participant_in_two_folds():
    df = _toy_df()
    folds = make_group_folds(df, n_splits=5, seed=0)
    assert len(folds) == len(df)

    per_participant_folds = pd.Series(folds).groupby(df["participant_id"]).nunique()
    assert (per_participant_folds == 1).all()


def test_folds_cover_all_rows_exactly_once():
    df = _toy_df()
    folds = make_group_folds(df, n_splits=5, seed=1)
    assert set(np.unique(folds)) <= set(range(5))
    assert (folds >= 0).all()


def test_make_group_folds_missing_columns_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        make_group_folds(df)


# ==============================================================================
# run_oof
# ==============================================================================


def test_oof_covers_every_video_exactly_once():
    df = _toy_df(n_participants=30, videos_per_participant=2, seed=2)

    def model_factory(df_train: pd.DataFrame):
        mean_x = df_train.groupby("label")["x"].mean()

        def predict(df_test: pd.DataFrame) -> np.ndarray:
            # Score = proximidade à média da classe 1 vs classe 0 (toy).
            m1 = mean_x.get(1, 1.0)
            m0 = mean_x.get(0, 0.0)
            d0 = np.abs(df_test["x"].to_numpy() - m0)
            d1 = np.abs(df_test["x"].to_numpy() - m1)
            return d0 / (d0 + d1 + 1e-9)

        return predict

    oof = run_oof(df, model_factory, n_splits=5, seed=3)
    assert oof.shape[0] == len(df)
    assert not np.isnan(oof).any()
    assert np.all((oof >= 0.0) & (oof <= 1.0))


def test_oof_predict_fn_wrong_length_raises():
    df = _toy_df(n_participants=20, videos_per_participant=2, seed=4)

    def bad_factory(df_train: pd.DataFrame):
        def predict(df_test: pd.DataFrame) -> np.ndarray:
            return np.zeros(1)  # tamanho errado de propósito

        return predict

    with pytest.raises(ValueError):
        run_oof(df, bad_factory, n_splits=5, seed=5)


# ==============================================================================
# Métricas
# ==============================================================================


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    ours = macro_f1_at_threshold(y_true, y_score, threshold=0.5)
    y_pred = (y_score >= 0.5).astype(int)
    expected = f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)
    assert ours == pytest.approx(expected)


def test_average_precision_matches_sklearn():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    ours = average_precision_score_(y_true, y_score)
    expected = average_precision_score(y_true, y_score)
    assert ours == pytest.approx(expected)


def test_average_precision_single_class_returns_zero():
    y_true = np.zeros(10, dtype=int)
    y_score = np.random.default_rng(2).random(10)
    assert average_precision_score_(y_true, y_score) == 0.0


# ==============================================================================
# Bootstrap
# ==============================================================================


def test_bootstrap_ci_macro_f1_contains_point_estimate():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=150)
    y_score = rng.random(150)
    point = macro_f1_at_threshold(y_true, y_score)
    result = bootstrap_ci_macro_f1(y_true, y_score, n_boot=300, seed=0)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    # A média do bootstrap deve estar perto da estimativa pontual.
    assert abs(result["mean"] - point) < 0.15


def test_paired_bootstrap_identical_scores_zero_diff():
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, size=150)
    y_score = rng.random(150)
    result = paired_bootstrap_macro_f1(y_true, y_score, y_score, n_boot=200, seed=0)
    assert result["observed_diff"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_paired_bootstrap_detects_clear_improvement():
    rng = np.random.default_rng(5)
    n = 300
    y_true = rng.integers(0, 2, size=n)
    # score_a é quase perfeito; score_b é ruído puro.
    score_a = np.where(y_true == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n))
    score_b = rng.random(n)
    result = paired_bootstrap_macro_f1(y_true, score_a, score_b, n_boot=300, seed=0)
    assert result["observed_diff"] > 0.0
    assert result["ci_low"] > 0.0
    assert result["p_value"] < 0.05
