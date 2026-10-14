"""Estratégias de limiar: escolhido SÓ na val, aplicado UMA vez no test (sem re-treinar).

Lê ``eval_val/predictions.csv`` e ``eval_test/predictions.csv`` de cada run ou ensemble
(``make evaluate``/``make ensemble-evaluate`` com ``SPLIT=val`` e ``SPLIT=test``) e compara:

- ``fixo_0.5``: sem calibração (a régua do protocolo OOF);
- ``argmax``: pico cru da curva F1×limiar na val;
- ``smooth``: centro do platô suavizado — a do artigo (``make reproduce-best``);
- ``base_rate``: prevê na val a mesma fração de positivos da val (quantil);
- ``bootstrap``: mediana do pico em reamostras da val (``aggregation.calibration=bootstrap``).

As estratégias são fixadas antes de olhar o test: a tabela compara todas, não escolhe "a
melhor no test". Grava ``<run>/threshold_sweep.json`` e imprime uma linha por run.

Uso:
    uv run python scripts/threshold_sweep.py outputs/ensemble/oficial-exp0 outputs/ensemble/...
    uv run python scripts/threshold_sweep.py --val-csv <val.csv> --test-csv <test.csv>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.aggregation import calibrate_threshold_from_video_scores
from src.training.metrics import average_precision, video_macro_f1

STRATEGIES = ["fixo_0.5", "argmax", "smooth", "base_rate", "bootstrap"]


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(np.int64), df["y_proba"].to_numpy(np.float32)


def sweep(val_csv: Path, test_csv: Path) -> dict:
    """Limiar de cada estratégia na val → Macro-F1 e fração prevista positiva no test."""
    val_y, val_p = _load(val_csv)
    test_y, test_p = _load(test_csv)
    results = []
    for name in STRATEGIES:
        thr = 0.5
        if name != "fixo_0.5":
            thr, _ = calibrate_threshold_from_video_scores(val_p, val_y, selection=name)
        pred_test = (test_p >= thr).astype(np.int64)
        results.append(
            {
                "strategy": name,
                "threshold": float(thr),
                "val_f1": video_macro_f1(val_y, (val_p >= thr).astype(np.int64)),
                "test_f1": video_macro_f1(test_y, pred_test),
                "test_pos_rate_pred": float(pred_test.mean()),
            }
        )
    return {
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "n_val": len(val_y),
        "n_test": len(test_y),
        "test_pos_rate_true": float(test_y.mean()),
        "test_ap": average_precision(test_y, test_p),
        "results": results,
    }


def _print_table(rows: list[tuple[str, dict]]) -> None:
    print("Macro-F1 no test [limiar escolhido na val]")
    print(f"{'run':<32} {'AP test':>8} " + " ".join(f"{s:>15}" for s in STRATEGIES))
    for name, rep in rows:
        cells = " ".join(
            f"{r['test_f1']:.4f} [{r['threshold']:.2f}]".rjust(15) for r in rep["results"]
        )
        print(f"{name:<32} {rep['test_ap']:>8.4f} {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estratégias de limiar val→test")
    parser.add_argument("runs", nargs="*", type=Path, help="pastas com eval_val/ e eval_test/")
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="JSON do modo --val-csv")
    args = parser.parse_args()

    pairs: list[tuple[str, Path, Path, Path]] = []
    if args.val_csv and args.test_csv:
        out = args.out or args.val_csv.parent / "threshold_sweep.json"
        pairs.append((str(args.val_csv.parent.parent), args.val_csv, args.test_csv, out))
    for run in args.runs:
        val, test = run / "eval_val" / "predictions.csv", run / "eval_test" / "predictions.csv"
        pairs.append((str(run).removeprefix("outputs/"), val, test, run / "threshold_sweep.json"))
    if not pairs:
        parser.error("passe pastas de runs ou --val-csv/--test-csv")

    rows = []
    for name, val, test, out in pairs:
        rep = sweep(val, test)
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append((name, rep))
    _print_table(rows)


if __name__ == "__main__":
    main()
