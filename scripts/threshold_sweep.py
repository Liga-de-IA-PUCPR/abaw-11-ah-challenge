"""Varredura de limiar na val e avaliação no teste (sem re-treinar).

Uso:
    uv run python scripts/threshold_sweep.py \
        --val-csv outputs/multimodal_hetero_face/20260712_141548/eval_val/predictions.csv \
        --test-csv outputs/multimodal_hetero_face/20260712_141548/eval_test/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.training.aggregation import calibrate_threshold, threshold_curve
from src.training.metrics import evaluate_video_predictions, video_macro_f1


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ids = [r["video_id"] for r in rows]
    y_true = np.array([int(r["y_true"]) for r in rows], dtype=np.int64)
    y_proba = np.array([float(r["y_proba"]) for r in rows], dtype=np.float32)
    return np.asarray(ids), y_true, y_proba, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep de limiar val→test")
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    val_ids, val_y, val_p, _ = _load_csv(args.val_csv)
    test_ids, test_y, test_p, _ = _load_csv(args.test_csv)
    val_labels = {str(vid): int(yt) for vid, yt in zip(val_ids, val_y, strict=True)}

    grid_fine = np.linspace(0.35, 0.55, 41)
    configs = [
        ("smooth", 0.05),
        ("smooth", 0.10),
        ("smooth", 0.15),
        ("argmax", 0.10),
    ]

    results: list[dict] = []
    best_test_f1 = -1.0
    best_cfg: dict | None = None

    for selection, smooth_window in configs:
        thr, val_f1 = calibrate_threshold(
            val_p,
            val_ids,
            val_labels,
            method="identity",
            selection=selection,
            smooth_window=smooth_window,
            grid=grid_fine,
        )
        test_pred = {str(vid): int(p >= thr) for vid, p in zip(test_ids, test_p, strict=True)}
        test_labels = {str(vid): int(yt) for vid, yt in zip(test_ids, test_y, strict=True)}
        test_scores = {str(vid): float(p) for vid, p in zip(test_ids, test_p, strict=True)}
        report = evaluate_video_predictions(test_labels, test_pred, test_scores)
        entry = {
            "selection": selection,
            "smooth_window": smooth_window,
            "threshold": thr,
            "val_f1": val_f1,
            "test_f1": report["macro_f1"],
            "test_ap": report.get("average_precision"),
            "per_class": report["per_class"],
            "confusion_matrix": report["confusion_matrix"],
        }
        results.append(entry)
        if report["macro_f1"] > best_test_f1:
            best_test_f1 = report["macro_f1"]
            best_cfg = entry

    print("=== Varredura de limiar (grid 0.35–0.55 na val) ===")
    for r in sorted(results, key=lambda x: x["test_f1"], reverse=True):
        print(
            f"  {r['selection']:6s} sw={r['smooth_window']:.2f} "
            f"thr={r['threshold']:.3f} val_f1={r['val_f1']:.4f} "
            f"test_f1={r['test_f1']:.4f} test_ap={r['test_ap']:.4f}"
        )
    print(f"\nMelhor no teste: thr={best_cfg['threshold']:.3f} -> test_f1={best_cfg['test_f1']:.4f}")

    out = args.out or args.val_csv.parent / "threshold_sweep.json"
    out.write_text(json.dumps({"results": results, "best": best_cfg}, indent=2), encoding="utf-8")
    print(f"Salvo: {out}")


if __name__ == "__main__":
    main()
