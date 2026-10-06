#!/usr/bin/env python3
"""(Re)gera ``plots/`` de runs OOF/route já gravados — os de antes do ``mode=oof`` gerar plots.

Lê ``oof_predictions.csv`` (ou ``route_oof.csv``) e o τ do ``*_metrics.json``; grava em
``<run>/plots/`` a matriz de confusão com τ, ROC, PR e a curva limiar × Macro-F1. Nada é
re-treinado::

    uv run python scripts/oof_plots.py outputs/oof/moe-r1-text/<ts> outputs/oof/moe-r5-face/<ts>
    make oof-plots RUN_DIR="outputs/oof/moe-r1-text/<ts> outputs/oof/moe-r5-face/<ts>"
    make oof-plots     # sem RUN_DIR: todo run em outputs/oof/*/* e outputs/route/*/* sem plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.oof import plot_run  # noqa: E402


def _pending(root: Path) -> list[Path]:
    """Runs OOF/route sob ``root`` com predições gravadas e sem ``plots/``."""
    runs = [p.parent for p in root.glob("oof/*/*/oof_predictions.csv")]
    runs += [p.parent for p in root.glob("route/*/*/route_oof.csv")]
    return sorted(r for r in runs if not (r / "plots").exists())


def main() -> None:
    parser = argparse.ArgumentParser(description="(Re)gera plots/ de runs OOF/route gravados.")
    parser.add_argument("runs", nargs="*", help="run dirs (default: os sem plots/ em --root)")
    parser.add_argument("--root", default=str(ROOT / "outputs"), help="raiz dos outputs")
    args = parser.parse_args()
    runs = [Path(r) for r in args.runs] or _pending(Path(args.root))
    if not runs:
        print("Nenhum run OOF/route sem plots/.")
    for run in runs:
        print(f"{run} → {plot_run(run)}")


if __name__ == "__main__":
    main()
