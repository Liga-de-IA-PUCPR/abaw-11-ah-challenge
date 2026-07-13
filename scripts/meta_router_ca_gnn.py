#!/usr/bin/env python3
"""Roteador meta CA(7 seeds) ⊕ GNN — seleção conjunta na val, test 1×.

Pipeline (sem features de anotação; sem peek no test):

1. Multi-start: otimiza pesos dos seeds CA no **train** (max F1).
2. Para cada vetor de pesos + (C, tau) do logreg de discordância,
   mede F1 na **val** do sistema completo (CA reponderada + swap GNN).
3. Escolhe o melhor pela val; avalia **uma vez** no test.

Uso::

    uv run python scripts/meta_router_ca_gnn.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
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


def _best_thr(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(0.2, 0.8, 121):
        f1 = _macro_f1(y, (scores >= thr).astype(np.int64))
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _load_split(split: str):
    import csv

    mats: list[dict[str, float]] = []
    labels: dict[str, int] = {}
    for seed in SEEDS:
        path = ROOT / "outputs" / "cross_attention" / seed / f"eval_{split}" / "predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        d: dict[str, float] = {}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d[row["video_id"]] = float(row["y_proba"])
                labels[row["video_id"]] = int(row["y_true"])
        mats.append(d)
    ids = sorted(labels)
    ca = np.array([[mats[j][vid] for j in range(7)] for vid in ids], dtype=np.float32)

    gnn_path = ROOT / GNN_RUN / f"eval_{split}" / "predictions.csv"
    with gnn_path.open(encoding="utf-8") as f:
        gnn = {r["video_id"]: float(r["y_proba"]) for r in csv.DictReader(f)}
    pg = np.array([gnn[vid] for vid in ids], dtype=np.float32)
    y = np.array([labels[vid] for vid in ids], dtype=np.int64)

    q_path = ROOT / "outputs" / "cross_attention" / SEEDS[0] / f"eval_{split}" / "predictions.csv"
    with q_path.open(encoding="utf-8") as f:
        qtype = {r["video_id"]: str(r.get("question_type") or "?") for r in csv.DictReader(f)}
    return ids, y, ca, pg, qtype


def _candidate_weights(ca_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)

    def train_obj(z: np.ndarray) -> float:
        w = _softmax(z)
        sc = (ca_tr * w).sum(1)
        return -_best_thr(sc, y_tr)[1]

    starts = (
        [np.zeros(7)]
        + [rng.normal(0, 0.5, size=7) for _ in range(20)]
        + [rng.normal(0, 1.0, size=7) for _ in range(10)]
    )
    weights: list[np.ndarray] = []
    for z0 in starts:
        res = minimize(train_obj, z0, method="Nelder-Mead", options={"maxiter": 350})
        w = _softmax(res.x)
        if not any(np.allclose(w, w2, atol=1e-3) for w2 in weights):
            weights.append(w)
    return weights


def _features(
    pca: np.ndarray,
    pg: np.ndarray,
    ids: list[str],
    qtype: dict[str, str],
    thr_ca: float,
    thr_gnn: float,
) -> np.ndarray:
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


def _pick_labels(
    pca: np.ndarray,
    pg: np.ndarray,
    y: np.ndarray,
    thr_ca: float,
    thr_gnn: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (pca >= thr_ca) != (pg >= thr_gnn)
    ca_ok = (pca >= thr_ca) == y
    g_ok = (pg >= thr_gnn) == y
    pick = np.full(len(y), -1, dtype=np.int64)
    for i in np.where(mask)[0]:
        if ca_ok[i] and not g_ok[i]:
            pick[i] = 0
        elif g_ok[i] and not ca_ok[i]:
            pick[i] = 1
        else:
            pick[i] = 0 if abs(pca[i] - thr_ca) >= abs(pg[i] - thr_gnn) else 1
    return mask, pick


def _route(
    clf: LogisticRegression,
    X: np.ndarray,
    pca: np.ndarray,
    pg: np.ndarray,
    thr_ca: float,
    thr_gnn: float,
    tau: float,
) -> np.ndarray:
    agree = (pca >= thr_ca) == (pg >= thr_gnn)
    pred = (pca >= thr_ca).astype(np.int64)
    if (~agree).any():
        p_gnn = clf.predict_proba(X[~agree])[:, 1]
        use_g = p_gnn >= tau
        pred = pred.copy()
        pred[~agree] = np.where(
            use_g,
            (pg[~agree] >= thr_gnn).astype(np.int64),
            (pca[~agree] >= thr_ca).astype(np.int64),
        )
    return pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr-gnn", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/ensemble_eval/meta_router_ca_gnn.json")
    args = parser.parse_args()

    ids_tr, y_tr, ca_tr, pg_tr, qt_tr = _load_split("train")
    ids_v, y_v, ca_v, pg_v, qt_v = _load_split("val")
    ids_t, y_t, ca_t, pg_t, qt_t = _load_split("test")

    thr_gnn = args.thr_gnn
    if thr_gnn is None:
        state = json.loads((ROOT / GNN_RUN / "trainer_state.json").read_text(encoding="utf-8"))
        thr_gnn = float(state["threshold"])

    weight_cands = _candidate_weights(ca_tr, y_tr, seed=args.seed)
    print(f"weight candidates: {len(weight_cands)}")

    best: dict | None = None
    for w in weight_cands:
        pca_tr = (ca_tr * w).sum(1)
        pca_v = (ca_v * w).sum(1)
        thr_ca, _ = _best_thr(pca_v, y_v)
        X_tr = _features(pca_tr, pg_tr, ids_tr, qt_tr, thr_ca, thr_gnn)
        X_v = _features(pca_v, pg_v, ids_v, qt_v, thr_ca, thr_gnn)
        mask_tr, pick_tr = _pick_labels(pca_tr, pg_tr, y_tr, thr_ca, thr_gnn)
        if int(mask_tr.sum()) < 10 or len(np.unique(pick_tr[mask_tr])) < 2:
            continue
        ca_alone_v = _macro_f1(y_v, (pca_v >= thr_ca).astype(np.int64))
        for C in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
            clf.fit(X_tr[mask_tr], pick_tr[mask_tr])
            for tau in np.linspace(0.30, 0.70, 17):
                pred_v = _route(clf, X_v, pca_v, pg_v, thr_ca, thr_gnn, float(tau))
                val_f1 = _macro_f1(y_v, pred_v)
                n_swap = int((pred_v != (pca_v >= thr_ca)).sum())
                cand = {
                    "val_f1": val_f1,
                    "n_swap_val": n_swap,
                    "C": float(C),
                    "tau": float(tau),
                    "thr_ca": float(thr_ca),
                    "w": w,
                    "clf": clf,
                    "ca_alone_v": ca_alone_v,
                }
                if best is None or val_f1 > best["val_f1"] or (
                    abs(val_f1 - best["val_f1"]) < 1e-12 and n_swap > best["n_swap_val"]
                ):
                    best = cand

    if best is None:
        raise RuntimeError("Nenhum candidato válido para o roteador")

    w = best["w"]
    thr_ca = best["thr_ca"]
    clf = best["clf"]
    tau = best["tau"]

    pca_t = (ca_t * w).sum(1)
    pca_v = (ca_v * w).sum(1)
    X_t = _features(pca_t, pg_t, ids_t, qt_t, thr_ca, thr_gnn)
    X_v = _features(pca_v, pg_v, ids_v, qt_v, thr_ca, thr_gnn)

    pred_t = _route(clf, X_t, pca_t, pg_t, thr_ca, thr_gnn, tau)
    labels = {vid: int(y) for vid, y in zip(ids_t, y_t, strict=True)}
    preds = {vid: int(p) for vid, p in zip(ids_t, pred_t, strict=True)}
    agree = (pca_t >= thr_ca) == (pg_t >= thr_gnn)
    scores_arr = pca_t.copy()
    n_swap_test = 0
    if (~agree).any():
        p_gnn = clf.predict_proba(X_t[~agree])[:, 1]
        use_g = p_gnn >= tau
        n_swap_test = int(use_g.sum())
        scores_arr = scores_arr.copy()
        scores_arr[~agree] = np.where(use_g, pg_t[~agree], pca_t[~agree])
    scores = {vid: float(s) for vid, s in zip(ids_t, scores_arr, strict=True)}
    report = evaluate_video_predictions(labels, preds, scores)

    out = {
        "method": "joint_val_select(ca_seed_weights + disagree_logreg)",
        "ca_seed_weights": w.tolist(),
        "thr_ca": thr_ca,
        "thr_gnn": thr_gnn,
        "logreg_C": best["C"],
        "tau_pick_gnn": tau,
        "val_f1_router": best["val_f1"],
        "val_f1_ca_alone": best["ca_alone_v"],
        "n_swap_val": best["n_swap_val"],
        "n_swap_test": n_swap_test,
        "n_test_disagree": int((~agree).sum()),
        "test_f1": report["macro_f1"],
        "test_ca_alone": _macro_f1(y_t, (pca_t >= thr_ca).astype(np.int64)),
        "test_gnn_alone": _macro_f1(y_t, (pg_t >= thr_gnn).astype(np.int64)),
        "report": report,
        "note": (
            "Seleção conjunta na val. Sem certainty_ah/ah_duration (vazam rótulo). "
            "Oracle pick ~0.80 é teto teórico."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary = {k: out[k] for k in out if k != "report"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"TEST macro_f1={report['macro_f1']:.4f} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
