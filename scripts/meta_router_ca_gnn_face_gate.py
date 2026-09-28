#!/usr/bin/env python3
"""Meta-router CA⊕GNN + gate de margem face (override conservador).

1. Trava o roteador 2-membro de ``meta_router_ca_gnn.json`` (ou re-calcula).
2. Quando CA==GNN e face discorda, sobrescreve para face se::

       face_margin >= ca_margin + delta
       face_margin >= gamma
       ca_margin   <= max_bm

   com ``(delta, gamma, max_bm)`` escolhidos na **val** (exige val ≥ baseline).

Uso::

    uv run python scripts/meta_router_ca_gnn_face_gate.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.training.metrics import evaluate_video_predictions

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
FACE_RUN = "outputs/face_gnn_ts/20260713_201255"
ROUTER_JSON = ROOT / "outputs/ensemble_eval/meta_router_ca_gnn.json"
QTYPES = [
    "ambivalent",
    "hesitant",
    "negative",
    "neutral",
    "positive",
    "resistant",
    "willing",
]


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)

    def f1(c: int) -> float:
        tp = float(((y == c) & (pred == c)).sum())
        fp = float(((y != c) & (pred == c)).sum())
        fn = float(((y == c) & (pred != c)).sum())
        den = 2 * tp + fp + fn
        return 0.0 if den == 0 else (2 * tp / den)

    return 0.5 * (f1(0) + f1(1))


def _load_proba(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        return {r["video_id"]: float(r["y_proba"]) for r in csv.DictReader(f)}


def _load_split(split: str, w: np.ndarray, face_run: Path):
    mats: list[dict[str, float]] = []
    labels: dict[str, int] = {}
    qtype: dict[str, str] = {}
    for i, seed in enumerate(SEEDS):
        path = ROOT / "outputs/cross_attention" / seed / f"eval_{split}/predictions.csv"
        d: dict[str, float] = {}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d[row["video_id"]] = float(row["y_proba"])
                labels[row["video_id"]] = int(row["y_true"])
                if i == 0:
                    qtype[row["video_id"]] = str(row.get("question_type") or "?")
        mats.append(d)
    ids = sorted(labels)
    ca = np.array([[mats[j][vid] for j in range(7)] for vid in ids], dtype=np.float64)
    pca = (ca * w).sum(1)
    gnn = _load_proba(ROOT / GNN_RUN / f"eval_{split}/predictions.csv")
    face = _load_proba(ROOT / face_run / f"eval_{split}/predictions.csv")
    pg = np.array([gnn[vid] for vid in ids], dtype=np.float64)
    pf = np.array([face[vid] for vid in ids], dtype=np.float64)
    y = np.array([labels[vid] for vid in ids], dtype=np.int64)
    return ids, y, pca, pg, pf, qtype


def _X_dis(pca, pg, ids, qtype, thr_ca, thr_gnn):
    rows = []
    for i, vid in enumerate(ids):
        q = qtype.get(vid, "?")
        qoh = [1.0 if q == t else 0.0 for t in QTYPES]
        rows.append(
            [
                float(pca[i]),
                float(pg[i]),
                abs(float(pca[i]) - thr_ca),
                abs(float(pg[i]) - thr_gnn),
                float(pca[i] - pg[i]),
                abs(float(pca[i] - pg[i])),
                float((pca[i] >= thr_ca) == (pg[i] >= thr_gnn)),
                float(pca[i] >= thr_ca),
                float(pg[i] >= thr_gnn),
                max(float(pca[i]), 1.0 - float(pca[i])),
                max(float(pg[i]), 1.0 - float(pg[i])),
                *qoh,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _route2(clf, pca, pg, ids, qtype, thr_ca, thr_gnn, tau):
    X = _X_dis(pca, pg, ids, qtype, thr_ca, thr_gnn)
    pred = (pca >= thr_ca).astype(np.int64)
    score = pca.copy()
    dis = (pca >= thr_ca) != (pg >= thr_gnn)
    if dis.any():
        p = clf.predict_proba(X[dis])[:, 1]
        use = p >= tau
        idx = np.where(dis)[0]
        pred[idx] = np.where(use, (pg[idx] >= thr_gnn).astype(np.int64), pred[idx])
        score[idx] = np.where(use, pg[idx], pca[idx])
    return pred, score


def _face_gate(pred, pca, pg, pf, thr_ca, thr_gnn, thr_face, delta, gamma, max_bm):
    out = pred.copy()
    pred_f = (pf >= thr_face).astype(np.int64)
    agree = (pca >= thr_ca) == (pg >= thr_gnn)
    dis = pred != pred_f
    fm = np.abs(pf - thr_face)
    cm = np.abs(pca - thr_ca)
    gate = agree & dis & (fm >= cm + delta) & (fm >= gamma) & (cm <= max_bm)
    out[gate] = pred_f[gate]
    return out, gate


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--face-run", type=Path, default=Path(FACE_RUN))
    p.add_argument("--router-json", type=Path, default=ROUTER_JSON)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/ensemble_eval/meta_router_ca_gnn_face_gate.json",
    )
    args = p.parse_args()

    rj = json.loads(args.router_json.read_text(encoding="utf-8"))
    w = np.asarray(rj["ca_seed_weights"], dtype=np.float64)
    thr_ca = float(rj["thr_ca"])
    thr_gnn = float(rj["thr_gnn"])
    C = float(rj["logreg_C"])
    tau = float(rj["tau_pick_gnn"])
    thr_face = float(
        json.loads((ROOT / args.face_run / "trainer_state.json").read_text(encoding="utf-8"))[
            "threshold"
        ]
    )

    tr = _load_split("train", w, args.face_run)
    va = _load_split("val", w, args.face_run)
    te = _load_split("test", w, args.face_run)

    # Fit disagree LogReg on train (same recipe as production)
    Xtr = _X_dis(tr[2], tr[3], tr[0], tr[5], thr_ca, thr_gnn)
    mask = (tr[2] >= thr_ca) != (tr[3] >= thr_gnn)
    ca_ok = (tr[2] >= thr_ca) == tr[1]
    g_ok = (tr[3] >= thr_gnn) == tr[1]
    pick = np.full(len(tr[1]), -1, dtype=np.int64)
    for i in np.where(mask)[0]:
        if ca_ok[i] and not g_ok[i]:
            pick[i] = 0
        elif g_ok[i] and not ca_ok[i]:
            pick[i] = 1
        else:
            pick[i] = 0 if abs(tr[2][i] - thr_ca) >= abs(tr[3][i] - thr_gnn) else 1
    clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
    clf.fit(Xtr[mask], pick[mask])

    pred_v, sc_v = _route2(clf, va[2], va[3], va[0], va[5], thr_ca, thr_gnn, tau)
    pred_t, sc_t = _route2(clf, te[2], te[3], te[0], te[5], thr_ca, thr_gnn, tau)
    base_v = _macro_f1(va[1], pred_v)
    base_t = _macro_f1(te[1], pred_t)

    best = None
    for delta in np.linspace(0.0, 0.2, 21):
        for gamma in np.linspace(0.05, 0.25, 21):
            for max_bm in np.linspace(0.05, 0.25, 21):
                pv, gv = _face_gate(
                    pred_v, va[2], va[3], va[4], thr_ca, thr_gnn, thr_face, delta, gamma, max_bm
                )
                vf1 = _macro_f1(va[1], pv)
                if vf1 + 1e-12 < base_v:
                    continue
                cand = {
                    "val_f1": vf1,
                    "delta": float(delta),
                    "gamma": float(gamma),
                    "max_bm": float(max_bm),
                    "n_swap_val": int(gv.sum()),
                }
                if best is None or vf1 > best["val_f1"] or (
                    abs(vf1 - best["val_f1"]) < 1e-12
                    and cand["n_swap_val"] > best["n_swap_val"]
                ):
                    best = cand

    if best is None:
        # identity fallback
        best = {"val_f1": base_v, "delta": 1.0, "gamma": 1.0, "max_bm": 0.0, "n_swap_val": 0}

    pred_final, gate_t = _face_gate(
        pred_t,
        te[2],
        te[3],
        te[4],
        thr_ca,
        thr_gnn,
        thr_face,
        best["delta"],
        best["gamma"],
        best["max_bm"],
    )
    scores = sc_t.copy()
    scores[gate_t] = te[4][gate_t]
    labels = {vid: int(y) for vid, y in zip(te[0], te[1], strict=True)}
    preds = {vid: int(p) for vid, p in zip(te[0], pred_final, strict=True)}
    sc = {vid: float(s) for vid, s in zip(te[0], scores, strict=True)}
    report = evaluate_video_predictions(labels, preds, sc)

    out = {
        "method": "meta_router_2mem + face_margin_gate",
        "face_run": str(args.face_run),
        "ca_seed_weights": w.tolist(),
        "thr_ca": thr_ca,
        "thr_gnn": thr_gnn,
        "thr_face": thr_face,
        "logreg_C": C,
        "tau_pick_gnn": tau,
        "gate": {
            "delta": best["delta"],
            "gamma": best["gamma"],
            "max_bm": best["max_bm"],
        },
        "val_f1_2mem": base_v,
        "val_f1_router": best["val_f1"],
        "n_swap_face_val": best["n_swap_val"],
        "n_swap_face_test": int(gate_t.sum()),
        "test_f1_2mem": base_t,
        "test_f1": report["macro_f1"],
        "report": report,
        "promote": bool(report["macro_f1"] > base_t),
        "note": (
            "Override face só se CA==GNN, face discorda, "
            "face_margin>=ca_margin+delta, face_margin>=gamma, ca_margin<=max_bm. "
            "Hiperparâmetros do gate selecionados na val."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary = {k: out[k] for k in out if k != "report"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"TEST macro_f1={report['macro_f1']:.4f} (2mem={base_t:.4f}, "
        f"Δ={report['macro_f1'] - base_t:+.4f}) → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
