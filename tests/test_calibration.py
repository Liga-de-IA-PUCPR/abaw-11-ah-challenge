"""Testes da calibração do limiar (seleção ``bootstrap``) e da varredura val→test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.aggregation import (
    THRESHOLD_STRATEGIES,
    _macro_f1_grid,
    calibrate_threshold_from_video_scores,
    compare_threshold_strategies,
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


def test_compare_strategies_picks_on_calibration_and_scores_on_eval():
    cal_y, cal_p = _noisy(seed=1)
    ev_y, ev_p = _noisy(n=300, seed=2)
    cmp = compare_threshold_strategies(cal_y, cal_p, ev_y, ev_p)
    assert [r["strategy"] for r in cmp["strategies"]] == list(THRESHOLD_STRATEGIES)
    for r in cmp["strategies"]:
        pred = (ev_p >= r["threshold"]).astype(np.int64)
        assert r["eval_f1"] == video_macro_f1(ev_y, pred)
        assert r["eval_pos_rate_pred"] == pred.mean()
    fixed = cmp["strategies"][0]
    assert fixed["strategy"] == "fixo_0.5" and fixed["threshold"] == 0.5
    assert cmp["n_calib"] == 124 and cmp["n_eval"] == 300


def test_evaluate_on_test_writes_the_threshold_comparison(compose_cfg):
    """mode=evaluate num split ≠ calibração grava eval_test/thresholds.json (caminho RF)."""
    import main
    from src.conf import resolve_device

    overrides = ("model=random_forest", "trainer=sklearn", "model.n_estimators=20")
    device = resolve_device("cpu")
    assert main._run_train(compose_cfg(*overrides), device) == 0
    run = next(Path(compose_cfg().data.paths.output_root).glob("random_forest/*/"))
    main._run_evaluate(compose_cfg(*overrides, "split=test", f"checkpoint={run}"), device)
    cmp = json.loads((run / "eval_test" / "thresholds.json").read_text())
    assert [r["strategy"] for r in cmp["strategies"]] == list(THRESHOLD_STRATEGIES)
    main._run_evaluate(compose_cfg(*overrides, "split=val", f"checkpoint={run}"), device)
    assert not (run / "eval_val" / "thresholds.json").exists()  # val é o de calibração


def test_threshold_table_reads_evaluated_runs_and_old_csvs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "threshold_sweep.py"
    old = tmp_path / "old_run"
    _write_preds(old / "eval_val", *_noisy(seed=1))
    _write_preds(old / "eval_test", *_noisy(n=300, seed=2))
    val_csv, test_csv = old / "eval_val/predictions.csv", old / "eval_test/predictions.csv"
    out = subprocess.run(
        [sys.executable, str(script), "--val-csv", str(val_csv), "--test-csv", str(test_csv),
         str(old)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert (old / "eval_test" / "thresholds.json").exists()  # o modo CSV grava, o run lê
    assert out.stdout.count("old_run") == 2 and "bootstrap" in out.stdout
