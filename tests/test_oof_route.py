"""Testes do runner OOF (mode=oof) e do MoERouter (mode=route) em dados sintéticos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.datasets import video_table
from src.pipeline.oof import _format_fold, run_oof


def test_format_fold_only_touches_placeholders():
    cfg = OmegaConf.create({"a": "runs/fold{fold}", "b": {"c": ["x{fold}", 3]}, "d": 1.5})
    out = _format_fold(cfg, 2)
    assert out.a == "runs/fold2" and list(out.b.c) == ["x2", 3] and out.d == 1.5


def test_oof_sklearn_covers_every_video_once_and_gates(compose_cfg, window_parquet):
    overrides = (
        "model=random_forest", "trainer=sklearn", "model.n_estimators=20", "oof.n_splits=3",
        "oof.n_boot=20", "experiment_name=rf-oof",
    )  # fmt: skip
    first = Path(run_oof(compose_cfg(*overrides))["out_dir"])
    preds = pd.read_csv(first / "oof_predictions.csv")
    videos = video_table(window_parquet, ["train", "val"])
    assert sorted(preds["video_id"]) == sorted(videos["video_id"])
    assert preds["y_proba"].between(0, 1).all() and set(preds["fold"]) == {0, 1, 2}
    # nenhum participante em duas dobras
    assert (preds.groupby("participant_id")["fold"].nunique() == 1).all()
    test = pd.read_csv(first / "pred_test.csv")
    assert len(test) == len(video_table(window_parquet, ["test"]))

    # mesmo modelo contra si mesmo → Δ = 0 → empate
    again = run_oof(compose_cfg(*overrides, f"oof.baseline={first}"))["out_dir"]
    gate = json.loads((Path(again) / "oof_metrics.json").read_text())["gate"]
    assert gate["macro_f1"]["observed_diff"] == pytest.approx(0.0)
    assert gate["verdict"] == "empate"


def _fake_member(root: Path, name: str, proba: np.ndarray, table: pd.DataFrame, emb_dim: int):
    run = root / name / "run"
    run.mkdir(parents=True)
    table.assign(y_proba=proba).to_csv(run / "oof_predictions.csv", index=False)
    rng = np.random.default_rng(len(name))
    if emb_dim:
        np.save(run / "oof_embeddings.npy", rng.normal(size=(len(table), emb_dim)).astype("f4"))
    test_ids = np.array([f"t{i}" for i in range(6)])
    arrays = {f"proba_fold{k}": rng.random(6) for k in range(3)}
    if emb_dim:
        arrays |= {f"embedding_fold{k}": rng.normal(size=(6, emb_dim)) for k in range(3)}
    np.savez(run / "pred_test_folds.npz", video_ids=test_ids, **arrays)
    return run


def test_route_combines_members_on_their_folds(compose_cfg, tmp_path):
    from src.pipeline.route import run_route

    rng = np.random.default_rng(0)
    n = 60
    y = rng.integers(0, 2, n)
    table = pd.DataFrame(
        {"video_id": [f"v{i}" for i in range(n)], "participant_id": [str(i // 2) for i in range(n)],
         "y_true": y, "fold": [(i // 2) % 3 for i in range(n)]}
    )  # fmt: skip
    good = np.clip(y * 0.6 + rng.random(n) * 0.4, 0.01, 0.99)
    noisy = rng.random(n)
    members = [
        _fake_member(tmp_path, "good", good, table, emb_dim=8),
        _fake_member(tmp_path, "noisy", noisy, table, emb_dim=0),  # só logit
    ]
    cfg = compose_cfg(
        f"route.members=[{members[0]},{members[1]}]", "route.n_boot=20", "route.max_epochs=50"
    )
    out = Path(run_route(cfg)["out_dir"])
    report = json.loads((out / "route_metrics.json").read_text())
    assert report["best_member"].startswith("good")
    assert sum(report["usage"].values()) == pytest.approx(1.0)
    assert len(pd.read_csv(out / "route_oof.csv")) == n
    assert len(pd.read_csv(out / "pred_test.csv")) == 6


def test_route_rejects_members_with_different_folds(compose_cfg, tmp_path):
    from src.pipeline.route import run_route

    table = pd.DataFrame({"video_id": ["a", "b"], "participant_id": ["1", "2"], "y_true": [0, 1],
                          "fold": [0, 1]})  # fmt: skip
    m1 = _fake_member(tmp_path, "m1", np.array([0.2, 0.8]), table, 0)
    m2 = _fake_member(tmp_path, "m2", np.array([0.3, 0.7]), table.assign(fold=[1, 0]), 0)
    with pytest.raises(ValueError, match="dobras"):
        run_route(compose_cfg(f"route.members=[{m1},{m2}]"))
