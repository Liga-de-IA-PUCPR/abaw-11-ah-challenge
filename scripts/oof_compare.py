#!/usr/bin/env python3
"""Gate pareado entre dois runs OOF (A − B): ΔMacro-F1@τ e ΔAP com IC por bootstrap pareado.

Compara quaisquer dois runs de ``mode=oof`` feitos com as MESMAS dobras (mesma ``oof.seed`` /
``oof.n_splits`` / ``oof.splits``) — ex.: a arquitetura MoE contra o cross-attention do artigo::

    uv run python scripts/oof_compare.py outputs/oof/moe-r5-face/<ts> \\
        outputs/oof/r0-cross-attention/<ts>
    make oof-compare A=outputs/oof/moe-r5-face/<ts> B=outputs/oof/r0-cross-attention/<ts>

Veredito: ``melhora`` se o IC 95% do ΔMacro-F1 fica todo acima de 0, ``piora`` se todo abaixo,
``empate`` caso contrário (o ΔAP vai junto como régua livre de limiar).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.oof import compare_runs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate pareado entre dois runs OOF (A − B).")
    parser.add_argument("run_a", help="run OOF avaliado (outputs/oof/<exp>/<ts>)")
    parser.add_argument("run_b", help="run OOF de referência")
    parser.add_argument("--threshold", type=float, default=0.5, help="τ fixo (default 0.5)")
    parser.add_argument("--n-boot", type=int, default=1000, help="reamostras do bootstrap")
    args = parser.parse_args()
    result = compare_runs(args.run_a, args.run_b, args.threshold, args.n_boot)
    f1, ap = result["macro_f1"], result["ap"]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"\nΔMacro-F1 = {f1['observed_diff']:+.4f} [{f1['ci_low']:+.4f}, {f1['ci_high']:+.4f}] "
        f"(p={f1['p_value']:.3f}) · ΔAP = {ap['observed_diff']:+.4f} "
        f"[{ap['ci_low']:+.4f}, {ap['ci_high']:+.4f}] → {result['verdict']} "
        f"({result['n_paired']} vídeos pareados)"
    )


if __name__ == "__main__":
    main()
