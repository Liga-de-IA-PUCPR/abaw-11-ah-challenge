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
    # o mesmo gate entre dois runs quaisquer, depois do fato (make oof-compare)
    from src.pipeline.oof import compare_runs

    pair = compare_runs(again, first, n_boot=20)
    assert pair["verdict"] == "empate" and pair["n_paired"] == len(videos)


def _fake_member(
    root: Path,
    name: str,
    proba: np.ndarray,
    table: pd.DataFrame,
    emb_dim: int,
    preds: dict[str, list[str]] | None = None,
):
    """Run OOF falso: ``oof_predictions.csv`` (+ embeddings) e, por nome em ``preds``, as
    saídas das 3 dobras (``pred_<nome>_folds.npz``) e a média (``pred_<nome>.csv``)."""
    run = root / name / "run"
    run.mkdir(parents=True)
    table.assign(y_proba=proba).to_csv(run / "oof_predictions.csv", index=False)
    rng = np.random.default_rng(len(name))
    if emb_dim:
        np.save(run / "oof_embeddings.npy", rng.normal(size=(len(table), emb_dim)).astype("f4"))
    for split, ids in (preds or {"test": [f"t{i}" for i in range(6)]}).items():
        arrays = {f"proba_fold{k}": rng.random(len(ids)) for k in range(3)}
        if emb_dim:
            arrays |= {f"embedding_fold{k}": rng.normal(size=(len(ids), emb_dim)) for k in range(3)}
        np.savez(run / f"pred_{split}_folds.npz", video_ids=np.array(ids), **arrays)
        mean = np.mean([arrays[f"proba_fold{k}"] for k in range(3)], axis=0)
        pd.DataFrame({"video_id": ids, "y_proba": mean}).to_csv(
            run / f"pred_{split}.csv", index=False
        )
    return run


def _oof_table(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    y = np.random.default_rng(seed).integers(0, 2, n)
    table = pd.DataFrame(
        {"video_id": [f"v{i}" for i in range(n)], "participant_id": [str(i // 2) for i in range(n)],
         "y_true": y, "fold": [(i // 2) % 3 for i in range(n)]}
    )  # fmt: skip
    return table, y


def test_route_combines_members_on_their_folds(compose_cfg, tmp_path):
    from src.pipeline.route import run_route

    rng = np.random.default_rng(0)
    table, y = _oof_table()
    n = len(table)
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


def test_holdout_split_is_participant_wise(compose_cfg, window_parquet):
    from src.data.datasets import load_split, load_train_val

    cfg = compose_cfg("data.train_splits=[train,val,test]", "data.calib_split=holdout",
                      "data.holdout_frac=0.2")  # fmt: skip
    train, hold = load_train_val(cfg, family="sklearn")
    pool = video_table(window_parquet, ["train", "val", "test"])
    assert set(train.video_labels) | set(hold.video_labels) == set(pool["video_id"])
    assert not set(train.groups) & set(hold.groups)  # nenhum participante nos dois
    assert 0.1 < len(hold.video_labels) / len(pool) < 0.3
    assert set(load_split(cfg, "holdout", family="sklearn").video_labels) == set(hold.video_labels)


def test_oof_predicts_extra_parquets_with_every_fold(compose_cfg, tmp_path):
    from tests.conftest import write_window_parquet

    ext = write_window_parquet(tmp_path / "external.parquet", n_participants=6, seed=1)
    out = Path(
        run_oof(
            compose_cfg(
                "model=random_forest", "trainer=sklearn", "model.n_estimators=10",
                "oof.n_splits=3", "oof.n_boot=10", "oof.predict_splits=[]",
                f"oof.predict_parquets={{external:'{ext}'}}",
            )
        )["out_dir"]
    )  # fmt: skip
    pred = pd.read_csv(out / "pred_external.csv")
    assert sorted(pred["video_id"]) == sorted(
        video_table(ext, ["train", "val", "test"])["video_id"]
    )
    folds = np.load(out / "pred_external_folds.npz")
    assert {"proba_fold0", "proba_fold1", "proba_fold2"} <= set(folds.files)


def test_route_measures_public_test_once_and_writes_submission(
    compose_cfg, tmp_path, window_parquet
):
    from src.pipeline.route import run_route

    table, y = _oof_table()
    test_ids = list(video_table(window_parquet, ["test"])["video_id"])
    ext_ids = [f"Videos/ext/{i}.mp4" for i in range(5)]
    preds = {"test": test_ids, "external": ext_ids}
    members = [
        _fake_member(tmp_path, "a", np.clip(y * 0.7 + 0.15, 0, 1), table, 4, preds),
        _fake_member(tmp_path, "b", np.full(len(y), 0.5), table, 0, preds),
    ]
    reference = tmp_path / "trial-0.txt"
    reference.write_text("".join(f"{v},0\n" for v in reversed(ext_ids)))
    cfg = compose_cfg(
        f"route.members=[{members[0]},{members[1]}]", "route.n_boot=10", "route.max_epochs=30",
        "route.predict_splits=[test,external]", "route.measure_splits=[test]",
        "route.submit_split=external", f"submission_reference={reference}",
        "submission_probabilities=true",
    )  # fmt: skip
    out = Path(run_route(cfg)["out_dir"])
    report = json.loads((out / "route_metrics.json").read_text())
    measured = report["measure_test"]
    assert measured["n_videos"] == len(test_ids) and set(measured["members"]) == {"a/run", "b/run"}
    lines = Path(report["submission"]).read_text().splitlines()
    assert [ln.split(",")[0] for ln in lines] == list(reversed(ext_ids))  # ordem da referência
    assert all(len(ln.split(",")) == 4 for ln in lines)  # video_id,p0,p1,pred


def test_route_measure_requires_the_split_prediction(compose_cfg, tmp_path):
    from src.pipeline.route import run_route

    table, y = _oof_table()
    m = [_fake_member(tmp_path, k, np.full(len(y), 0.5), table, 0) for k in ("a", "b")]
    cfg = compose_cfg(f"route.members=[{m[0]},{m[1]}]", "route.max_epochs=5",
                      "route.measure_splits=[val]")  # fmt: skip
    with pytest.raises(ValueError, match="predict_splits"):
        run_route(cfg)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (["mode=oof", "+experiment=moe_r1_text"], "oof/moe-r1-text"),
        (["mode=oof", "experiment_name=exp0"], "oof/exp0"),
        (["mode=train", "+experiment=cross_attention"], "cross_attention"),
        (["mode=route"], "route/moe_router"),
        (["mode=featurize_columns"], "logs/featurize_columns"),
        (["mode=evaluate"], "logs/evaluate"),
    ],
)
def test_hydra_run_dir_is_the_run_folder_of_each_mode(overrides, expected):
    """O Hydra grava .hydra/ + main.log NA pasta onde o modo grava os artefatos."""
    from hydra import compose, initialize_config_dir

    from tests.conftest import CONFIGS

    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose("config", overrides=overrides, return_hydra_config=True)
        run_dir = Path(str(cfg.hydra.run.dir))
    assert run_dir.parent == Path("outputs") / expected
    assert len(run_dir.name) == len("20261006_120000")


def test_hydra_run_dir_is_none_outside_a_hydra_run():
    from src.outputs.checkpoint import hydra_run_dir

    assert hydra_run_dir() is None
