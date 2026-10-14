"""Tabela das estratégias de limiar (val → test) de vários runs — ``make thresholds``.

A comparação em si é do código: o ``mode=evaluate`` num split diferente do de calibração
grava ``eval_<split>/thresholds.json`` (``compare_threshold_strategies`` em
``src/training/aggregation.py``: cada estratégia escolhe o limiar SÓ no val e é aplicada
UMA vez no test). Este script só junta esses arquivos numa tabela — ou, com
``--val-csv/--test-csv``, compara direto dois ``predictions.csv`` já gravados (runs antigos).

Uso:
    make thresholds RUNS="outputs/ensemble/oficial-exp0 outputs/ensemble/oficial-exp1"
    uv run python scripts/threshold_sweep.py --val-csv <val.csv> --test-csv <test.csv>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.training.aggregation import THRESHOLD_STRATEGIES, compare_threshold_strategies


def _from_csvs(val_csv: Path, test_csv: Path) -> dict:
    val, test = pd.read_csv(val_csv), pd.read_csv(test_csv)
    return compare_threshold_strategies(
        val["y_true"], val["y_proba"], test["y_true"], test["y_proba"]
    )


def _print_table(rows: list[tuple[str, dict]], split: str) -> None:
    print(f"Macro-F1 no {split} [limiar escolhido no val]")
    print(f"{'run':<32} {'AP':>7} " + " ".join(f"{s:>15}" for s in THRESHOLD_STRATEGIES))
    for name, cmp in rows:
        cells = " ".join(
            f"{r['eval_f1']:.4f} [{r['threshold']:.2f}]".rjust(15) for r in cmp["strategies"]
        )
        print(f"{name:<32} {cmp['eval_ap']:>7.4f} {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estratégias de limiar val→test")
    parser.add_argument("runs", nargs="*", type=Path, help="runs/ensembles já avaliados")
    parser.add_argument("--split", default="test", help="split avaliado (eval_<split>/)")
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    args = parser.parse_args()

    rows: list[tuple[str, dict]] = []
    if args.val_csv and args.test_csv:
        cmp = _from_csvs(args.val_csv, args.test_csv)
        out = args.test_csv.parent / "thresholds.json"
        out.write_text(json.dumps(cmp, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append((str(args.test_csv.parent.parent), cmp))
    for run in args.runs:
        path = run / f"eval_{args.split}" / "thresholds.json"
        if not path.exists():
            raise SystemExit(f"{path} não existe: rode o evaluate com SPLIT={args.split}")
        rows.append((str(run).removeprefix("outputs/"), json.loads(path.read_text())))
    if not rows:
        parser.error("passe runs avaliados ou --val-csv/--test-csv")
    _print_table(rows, args.split)


if __name__ == "__main__":
    main()
