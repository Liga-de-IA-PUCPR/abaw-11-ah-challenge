"""Otimiza pesos de ensemble usando predições já exportadas.

Escolhe pesos e limiar somente na validação, depois reporta o teste uma vez.
Não carrega modelos nem usa GPU.

Exemplo:
    uv run python scripts/ensemble_sweep.py \
      --member full:outputs/.../full/eval_val/predictions.csv:outputs/.../full/eval_test/predictions.csv \
      --member gnn:outputs/.../gnn/eval_val/predictions.csv:outputs/.../gnn/eval_test/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from src.training.metrics import evaluate_video_predictions


def _read_preds(path: Path) -> dict[str, tuple[int, float]]:
    rows = csv.DictReader(path.open(encoding="utf-8"))
    return {r["video_id"]: (int(r["y_true"]), float(r["y_proba"])) for r in rows}


def _parse_member(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--member deve ser nome:val_csv:test_csv")
    return parts[0], Path(parts[1]), Path(parts[2])


def _weight_grid(n: int, step: float) -> list[tuple[float, ...]]:
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    combos_set: set[tuple[float, ...]] = set()
    for raw in itertools.product(vals, repeat=n):
        s = sum(raw)
        if s <= 0:
            continue
        norm = tuple(round(float(v / s), 6) for v in raw)
        combos_set.add(norm)
    return sorted(combos_set)


def _combine(
    preds: list[dict[str, tuple[int, float]]],
    weights: tuple[float, ...],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(set.intersection(*(set(p.keys()) for p in preds)))
    y_true = np.array([preds[0][vid][0] for vid in ids], dtype=np.int64)
    score = np.zeros(len(ids), dtype=np.float32)
    for pred, weight in zip(preds, weights, strict=True):
        score += float(weight) * np.array([pred[vid][1] for vid in ids], dtype=np.float32)
    return ids, y_true, score


def _macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    f1_pos = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    f1_neg = (2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    return 0.5 * (f1_pos + f1_neg)


def _best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.0, 1.0, 101)
    f1s = np.array([_macro_f1_fast(y_true, (score >= thr).astype(np.int64)) for thr in grid])
    idx = int(np.argmax(f1s))
    return float(grid[idx]), float(f1s[idx])


def _report(ids: list[str], y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    labels = {vid: int(y) for vid, y in zip(ids, y_true, strict=True)}
    pred = {vid: int(s >= threshold) for vid, s in zip(ids, score, strict=True)}
    scores = {vid: float(s) for vid, s in zip(ids, score, strict=True)}
    return evaluate_video_predictions(labels, pred, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("outputs/ensemble_eval/weight_sweep.json"))
    args = parser.parse_args()

    members = [_parse_member(m) for m in args.member]
    names = [m[0] for m in members]
    val_preds = [_read_preds(m[1]) for m in members]
    test_preds = [_read_preds(m[2]) for m in members]

    candidates: list[dict] = []
    for weights in _weight_grid(len(members), args.step):
        val_ids, val_y, val_score = _combine(val_preds, weights)
        threshold, val_f1 = _best_threshold(val_y, val_score)
        candidates.append(
            {
                "members": names,
                "weights": dict(zip(names, weights, strict=True)),
                "threshold": threshold,
                "val_f1": val_f1,
            }
        )

    results: list[dict] = []
    for row in sorted(candidates, key=lambda r: r["val_f1"], reverse=True)[:25]:
        weights = tuple(float(row["weights"][name]) for name in names)
        test_ids, test_y, test_score = _combine(test_preds, weights)
        test_report = _report(test_ids, test_y, test_score, float(row["threshold"]))
        results.append({
            "members": names,
            "weights": row["weights"],
            "threshold": row["threshold"],
            "val_f1": row["val_f1"],
            "test_f1": test_report["macro_f1"],
            "test_ap": test_report.get("average_precision", 0.0),
            "test_confusion_matrix": test_report["confusion_matrix"],
            "test_per_class": test_report["per_class"],
        })

    best = results[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"best_by_val": best, "top_by_val": results}, indent=2), encoding="utf-8"
    )

    print("=== Melhor por validação ===")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print("\n=== Top 10 por validação ===")
    for row in sorted(results, key=lambda r: r["val_f1"], reverse=True)[:10]:
        print(
            f"val={row['val_f1']:.4f} test={row['test_f1']:.4f} "
            f"thr={row['threshold']:.2f} weights={row['weights']}"
        )
    print(f"\nSalvo: {args.out}")


if __name__ == "__main__":
    main()
