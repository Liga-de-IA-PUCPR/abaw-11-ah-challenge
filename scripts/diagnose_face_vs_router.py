#!/usr/bin/env python3
"""Diagnóstico: face vs partições CA⊕GNN (both_wrong / rescues).

Reusa pesos/thresholds do meta-router CA⊕GNN se o JSON existir; senão equal-weight CA.

Uso::

    uv run python scripts/diagnose_face_vs_router.py \\
      --face-run outputs/face_gnn_ts/<run> \\
      --out outputs/ensemble_eval/face_vs_router_diag.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [
    "20260713_175242",
    "20260713_175257",
    "20260713_175312",
    "20260713_175327",
    "20260713_175342",
    "20260713_175357",
    "20260713_175412",
]
GNN_RUN = "outputs/hetero_gnn_contrastive/20260713_162753"
ROUTER_JSON = ROOT / "outputs/ensemble_eval/meta_router_ca_gnn.json"


def _load_csv(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["video_id"]] = row
    return out


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    def f1(c: int) -> float:
        tp = float(((y == c) & (pred == c)).sum())
        fp = float(((y != c) & (pred == c)).sum())
        fn = float(((y == c) & (pred != c)).sum())
        den = 2 * tp + fp + fn
        return 0.0 if den == 0 else (2 * tp / den)

    return 0.5 * (f1(0) + f1(1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--face-run", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--thr-face", type=float, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/ensemble_eval/face_vs_router_diag.json")
    args = p.parse_args()

    face_csv = ROOT / args.face_run / f"eval_{args.split}" / "predictions.csv"
    if not face_csv.exists():
        raise FileNotFoundError(face_csv)

    thr_face = args.thr_face
    if thr_face is None:
        state = json.loads((ROOT / args.face_run / "trainer_state.json").read_text(encoding="utf-8"))
        thr_face = float(state["threshold"])

    thr_ca, thr_gnn = 0.39, 0.34
    w = np.ones(7, dtype=np.float64) / 7.0
    if ROUTER_JSON.exists():
        rj = json.loads(ROUTER_JSON.read_text(encoding="utf-8"))
        thr_ca = float(rj["thr_ca"])
        thr_gnn = float(rj["thr_gnn"])
        w = np.asarray(rj["ca_seed_weights"], dtype=np.float64)

    face = _load_csv(face_csv)
    mats = []
    labels: dict[str, int] = {}
    for seed in SEEDS:
        path = ROOT / "outputs/cross_attention" / seed / f"eval_{args.split}/predictions.csv"
        d = _load_csv(path)
        mats.append(d)
        for vid, row in d.items():
            labels[vid] = int(row["y_true"])
    gnn = _load_csv(ROOT / GNN_RUN / f"eval_{args.split}/predictions.csv")

    ids = sorted(set(labels) & set(face) & set(gnn))
    y = np.array([labels[v] for v in ids], dtype=np.int64)
    pca = np.array(
        [sum(float(mats[j][v]["y_proba"]) * w[j] for j in range(7)) for v in ids],
        dtype=np.float64,
    )
    pg = np.array([float(gnn[v]["y_proba"]) for v in ids], dtype=np.float64)
    pf = np.array([float(face[v]["y_proba"]) for v in ids], dtype=np.float64)

    pred_ca = (pca >= thr_ca).astype(np.int64)
    pred_g = (pg >= thr_gnn).astype(np.int64)
    pred_f = (pf >= thr_face).astype(np.int64)

    ca_ok = pred_ca == y
    g_ok = pred_g == y
    f_ok = pred_f == y

    both_ok = ca_ok & g_ok
    ca_only = ca_ok & ~g_ok
    g_only = g_ok & ~ca_ok
    both_wrong = ~ca_ok & ~g_ok
    face_rescues = both_wrong & f_ok
    face_only_rescue = face_rescues  # same

    agree_cg = pred_ca == pred_g
    face_disagrees_agree = agree_cg & (pred_f != pred_ca)
    face_rescue_on_agree = face_disagrees_agree & f_ok & ~ca_ok

    out = {
        "split": args.split,
        "n": len(ids),
        "thr_ca": thr_ca,
        "thr_gnn": thr_gnn,
        "thr_face": thr_face,
        "face_run": str(args.face_run),
        "counts": {
            "both_ok": int(both_ok.sum()),
            "ca_only": int(ca_only.sum()),
            "gnn_only": int(g_only.sum()),
            "both_wrong": int(both_wrong.sum()),
            "face_rescues_both_wrong": int(face_rescues.sum()),
            "face_disagrees_when_ca_gnn_agree": int(face_disagrees_agree.sum()),
            "face_correct_on_agree_errors": int(face_rescue_on_agree.sum()),
            "face_alone_ok": int(f_ok.sum()),
        },
        "macro_f1": {
            "ca": _macro_f1(y, pred_ca),
            "gnn": _macro_f1(y, pred_g),
            "face": _macro_f1(y, pred_f),
            "oracle_ca_gnn_face": _macro_f1(
                y,
                np.where(
                    ca_ok,
                    pred_ca,
                    np.where(g_ok, pred_g, np.where(f_ok, pred_f, pred_ca)),
                ),
            ),
        },
        "note": (
            "face_rescues_both_wrong = CA e GNN erram e face acerta. "
            "Oracle pick entre os 3 (teto teórico)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
