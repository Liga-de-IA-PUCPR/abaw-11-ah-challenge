#!/usr/bin/env python3
"""Ensemble CA⊕GNN com calibração na val — gated / qtype-aware (não mean cego).

O mean weighted dilui a CA (0.724→0.716). Complementaridade real:
GNN salva ~42 vídeos que a CA erra e vice-versa ~44; em ~97 ambos erram
(sinal de dataset, não de miner). Este script:

1. Calibra limiar (e estratégia) **só na val**
2. Aplica no test uma vez
3. ``qtype_soft``: pesos por question_type aprendidos na val
4. ``gated``: mistura GNN só quando CA está perto do limiar

Hard mining (WeightedRandomSampler) e miner ``batch_hard`` (triplets no batch)
são **caminhos separados** — nenhum deles lê este ensemble.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.training.metrics import evaluate_video_predictions

ROOT = Path(__file__).resolve().parents[1]


def _read_preds(path: Path) -> dict[str, tuple[int, float]]:
    with path.open(encoding="utf-8") as f:
        return {r["video_id"]: (int(r["y_true"]), float(r["y_proba"])) for r in csv.DictReader(f)}


def _read_qtype(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "question_type" not in rows[0]:
        return {}
    return {r["video_id"]: str(r.get("question_type") or "?") for r in rows}


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    def f1(c: int) -> float:
        tp = float(((y == c) & (pred == c)).sum())
        fp = float(((y != c) & (pred == c)).sum())
        fn = float(((y == c) & (pred != c)).sum())
        den = 2 * tp + fp + fn
        return 0.0 if den == 0 else (2 * tp / den)

    return 0.5 * (f1(0) + f1(1))


def _best_thr(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(0.2, 0.8, 61):
        f1 = _macro_f1(y, (scores >= thr).astype(np.int64))
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def _combine(
    ca: dict[str, tuple[int, float]],
    gnn: dict[str, tuple[int, float]],
    qtype: dict[str, str],
    prefer: dict[str, str],
    mode: str,
    thr_ca: float,
    w_ca_default: float,
    w_ca_gnn_pref: float,
    margin: float,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(set(ca) & set(gnn))
    y = np.array([ca[v][0] for v in ids], dtype=np.int64)
    scores = np.zeros(len(ids), dtype=np.float32)
    for i, vid in enumerate(ids):
        pca = ca[vid][1]
        pg = gnn[vid][1]
        q = qtype.get(vid, "?")
        if mode == "ca":
            s = pca
        elif mode == "mean":
            s = w_ca_default * pca + (1.0 - w_ca_default) * pg
        elif mode == "gated":
            s = 0.5 * pca + 0.5 * pg if abs(pca - thr_ca) < margin else pca
        elif mode == "qtype_soft":
            if prefer.get(q) == "gnn":
                s = w_ca_gnn_pref * pca + (1.0 - w_ca_gnn_pref) * pg
            else:
                s = w_ca_default * pca + (1.0 - w_ca_default) * pg
        else:
            raise ValueError(mode)
        scores[i] = s
    return ids, y, scores


def _prefer_qtype(
    ca: dict[str, tuple[int, float]],
    gnn: dict[str, tuple[int, float]],
    qtype: dict[str, str],
    thr_ca: float,
    thr_gnn: float,
) -> dict[str, str]:
    stats: dict[str, Counter] = defaultdict(Counter)
    for vid in set(ca) & set(gnn) & set(qtype):
        yt, pca = ca[vid]
        _, pg = gnn[vid]
        q = qtype[vid]
        ca_ok = int(pca >= thr_ca) == yt
        g_ok = int(pg >= thr_gnn) == yt
        if ca_ok and not g_ok:
            stats[q]["ca"] += 1
        elif g_ok and not ca_ok:
            stats[q]["gnn"] += 1
    return {q: ("gnn" if c["gnn"] > c["ca"] else "ca") for q, c in stats.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ca-val", type=Path, default=ROOT / "outputs/ensemble_eval/_export/ca_val_mean.csv")
    p.add_argument("--ca-test", type=Path, default=ROOT / "outputs/ensemble_eval/_export/ca_test_mean.csv")
    p.add_argument(
        "--gnn-val",
        type=Path,
        default=ROOT / "outputs/hetero_gnn_contrastive/20260713_162753/eval_val/predictions.csv",
    )
    p.add_argument(
        "--gnn-test",
        type=Path,
        default=ROOT / "outputs/hetero_gnn_contrastive/20260713_162753/eval_test/predictions.csv",
    )
    p.add_argument(
        "--qtype-val",
        type=Path,
        default=ROOT / "outputs/cross_attention/20260713_175242/eval_val/predictions.csv",
    )
    p.add_argument(
        "--qtype-test",
        type=Path,
        default=ROOT / "outputs/cross_attention/20260713_175242/eval_test/predictions.csv",
    )
    p.add_argument("--thr-gnn", type=float, default=None)
    p.add_argument("--mode", choices=["ca", "mean", "gated", "qtype_soft"], default="qtype_soft")
    p.add_argument("--w-ca", type=float, default=0.85, help="peso CA quando prefer=ca")
    p.add_argument("--w-ca-when-gnn", type=float, default=0.35, help="peso CA quando prefer=gnn")
    p.add_argument("--margin", type=float, default=0.08)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/ensemble_eval/gated_ca_gnn.json")
    args = p.parse_args()

    ca_v, ca_t = _read_preds(args.ca_val), _read_preds(args.ca_test)
    gnn_v, gnn_t = _read_preds(args.gnn_val), _read_preds(args.gnn_test)
    qt_v, qt_t = _read_qtype(args.qtype_val), _read_qtype(args.qtype_test)

    thr_gnn = args.thr_gnn
    if thr_gnn is None:
        state = ROOT / "outputs/hetero_gnn_contrastive/20260713_162753/trainer_state.json"
        thr_gnn = float(json.loads(state.read_text(encoding="utf-8"))["threshold"])

    # thr_ca preliminar na val (CA só)
    ids = sorted(set(ca_v) & set(gnn_v))
    thr_ca, _ = _best_thr(
        np.array([ca_v[v][1] for v in ids], dtype=np.float32),
        np.array([ca_v[v][0] for v in ids], dtype=np.int64),
    )
    prefer = _prefer_qtype(ca_v, gnn_v, qt_v, thr_ca, thr_gnn)

    ids_v, y_v, sc_v = _combine(
        ca_v, gnn_v, qt_v, prefer, args.mode, thr_ca, args.w_ca, args.w_ca_when_gnn, args.margin
    )
    thr, val_f1 = _best_thr(sc_v, y_v)

    ids_t, y_t, sc_t = _combine(
        ca_t, gnn_t, qt_t, prefer, args.mode, thr_ca, args.w_ca, args.w_ca_when_gnn, args.margin
    )
    labels = {vid: int(y) for vid, y in zip(ids_t, y_t, strict=True)}
    preds = {vid: int(s >= thr) for vid, s in zip(ids_t, sc_t, strict=True)}
    scores = {vid: float(s) for vid, s in zip(ids_t, sc_t, strict=True)}
    report = evaluate_video_predictions(labels, preds, scores)

    out = {
        "mode": args.mode,
        "prefer_qtype": prefer,
        "thr_ca_ref": thr_ca,
        "thr_gnn": thr_gnn,
        "threshold": thr,
        "val_f1": val_f1,
        "test_f1": report["macro_f1"],
        "w_ca": args.w_ca,
        "w_ca_when_gnn": args.w_ca_when_gnn,
        "margin": args.margin,
        "report": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k != "report"}, indent=2, ensure_ascii=False))
    print(f"test macro_f1={report['macro_f1']:.4f} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
