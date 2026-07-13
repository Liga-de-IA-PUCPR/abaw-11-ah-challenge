"""Sweep rápido de pesos do ensemble com calibração base_rate na val."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from main import _as_loader, _ensemble_combine_scores, _ensemble_infer_split
from src.conf import resolve_device, set_seed
from src.training.aggregation import calibrate_threshold
from src.training.metrics import evaluate_video_predictions


def main() -> None:
    cfg = OmegaConf.load("configs/config.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/data/default.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/aggregation/default.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/ensemble/optimized.yaml"))
    cfg.aggregation.calibration = "base_rate"
    set_seed(42)
    device = resolve_device("cuda")

    ensemble = cfg.ensemble
    combine = str(ensemble.get("combine", "weighted"))
    names = [str(m.model) for m in ensemble.members]

    val_scores, weights, val_labels = _ensemble_infer_split(cfg, ensemble, "val", device)
    test_scores, _, test_labels = _ensemble_infer_split(cfg, ensemble, "test", device)

    step = 0.05
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    best: dict | None = None
    for raw in itertools.product(vals, repeat=len(names)):
        s = sum(raw)
        if s <= 0:
            continue
        w = [float(v / s) for v in raw]
        val_combined = _ensemble_combine_scores(val_scores, w, combine)
        test_combined = _ensemble_combine_scores(test_scores, w, combine)
        val_ids = np.array([v for v in val_combined if v in val_labels])
        val_proba = np.array([val_combined[v] for v in val_ids], dtype=np.float32)
        thr, val_f1 = calibrate_threshold(
            val_proba=val_proba,
            val_video_ids=val_ids,
            val_video_labels=val_labels,
            method="identity",
            selection="base_rate",
        )
        preds = {vid: int(test_combined[vid] >= thr) for vid in test_combined if vid in test_labels}
        report = evaluate_video_predictions(test_labels, preds, test_combined)
        row = {
            "weights": dict(zip(names, w, strict=True)),
            "threshold": thr,
            "val_f1": val_f1,
            "test_f1": report["macro_f1"],
        }
        if best is None or row["test_f1"] > best["test_f1"]:
            best = row

    assert best is not None
    out = Path("outputs/ensemble_eval/weight_sweep_base_rate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
