#!/usr/bin/env python3
"""Gera hard_examples.json a partir de predições — focado em hard pos/neg (não rare).

O hard_mining legado aplica rare_mult em ~40% do train e dilui o sinal dos 18 hard
reais. Este script:

- hard_pos: y=1 e proba < hard_lo  → × hard_weight
- hard_neg: y=0 e proba > hard_hi  → × hard_weight
- border: |proba-thr| < border       → × border_weight
- opcional: boost se CA e GNN erram juntos (both_wrong) quando --gnn-csv é passado
- rare_mult default = 1.0 (desligado)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read(path: Path) -> dict[str, tuple[int, float]]:
    with path.open(encoding="utf-8") as f:
        return {r["video_id"]: (int(r["y_true"]), float(r["y_proba"])) for r in csv.DictReader(f)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ca-csv", type=Path, required=True, help="predictions.csv (ex.: CA no train)")
    p.add_argument("--gnn-csv", type=Path, default=None, help="opcional — boost both_wrong")
    p.add_argument("--thr-ca", type=float, default=0.43)
    p.add_argument("--thr-gnn", type=float, default=0.34)
    p.add_argument("--hard-hi", type=float, default=0.65)
    p.add_argument("--hard-lo", type=float, default=0.35)
    p.add_argument("--hard-weight", type=float, default=4.0)
    p.add_argument("--border", type=float, default=0.08)
    p.add_argument("--border-weight", type=float, default=2.0)
    p.add_argument("--both-wrong-weight", type=float, default=1.5)
    p.add_argument("--out", type=Path, default=Path("data/interim/hard_examples_ca.json"))
    args = p.parse_args()

    ca = _read(args.ca_csv)
    gnn = _read(args.gnn_csv) if args.gnn_csv else {}

    weights: dict[str, float] = {}
    n_hard = n_border = n_both = 0
    for vid, (yt, p_ca) in ca.items():
        w = 1.0
        hard = (yt == 1 and p_ca < args.hard_lo) or (yt == 0 and p_ca > args.hard_hi)
        if hard:
            w *= args.hard_weight
            n_hard += 1
        elif abs(p_ca - args.thr_ca) < args.border:
            w *= args.border_weight
            n_border += 1
        if vid in gnn:
            _, p_g = gnn[vid]
            ca_wrong = int(p_ca >= args.thr_ca) != yt
            g_wrong = int(p_g >= args.thr_gnn) != yt
            if ca_wrong and g_wrong:
                w *= args.both_wrong_weight
                n_both += 1
        weights[vid] = round(w, 4)

    payload = {
        "source": "mine_hard_from_preds",
        "ca_csv": str(args.ca_csv),
        "gnn_csv": str(args.gnn_csv) if args.gnn_csv else None,
        "params": {
            "hard_hi": args.hard_hi,
            "hard_lo": args.hard_lo,
            "hard_weight": args.hard_weight,
            "border": args.border,
            "border_weight": args.border_weight,
            "both_wrong_weight": args.both_wrong_weight,
            "rare_mult": 1.0,
        },
        "n_videos": len(weights),
        "n_hard": n_hard,
        "n_border": n_border,
        "n_both_wrong": n_both,
        "weights": weights,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"hard={n_hard} border={n_border} both_wrong={n_both} → {args.out} "
        f"(peso∈[{min(weights.values()):.1f}, {max(weights.values()):.1f}])"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
