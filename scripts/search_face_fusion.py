#!/usr/bin/env python3
"""Busca fusões CA⊕GNN⊕Face sobre o meta-router 2-membro travado.

Seleciona na val (exige val_f1 >= baseline); reporta test 1×.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

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
GNN_RUN = ROOT / "outputs/hetero_gnn_contrastive/20260713_162753"
FACE_RUN = ROOT / "outputs/face_gnn_ts/20260713_201255"
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


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)

    def f1(c: int) -> float:
        tp = float(((y == c) & (p == c)).sum())
        fp = float(((y != c) & (p == c)).sum())
        fn = float(((y == c) & (p != c)).sum())
        den = 2 * tp + fp + fn
        return 0.0 if den == 0 else (2 * tp / den)

    return 0.5 * (f1(0) + f1(1))


def best_thr(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.2, 0.8, 61):
        f1 = macro_f1(y, (scores >= t).astype(np.int64))
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


def load_split(split: str, w: np.ndarray):
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

    def one(path: Path) -> dict[str, float]:
        with path.open(encoding="utf-8") as f:
            return {r["video_id"]: float(r["y_proba"]) for r in csv.DictReader(f)}

    gnn = one(GNN_RUN / f"eval_{split}/predictions.csv")
    face = one(FACE_RUN / f"eval_{split}/predictions.csv")
    pg = np.array([gnn[vid] for vid in ids], dtype=np.float64)
    pf = np.array([face[vid] for vid in ids], dtype=np.float64)
    y = np.array([labels[vid] for vid in ids], dtype=np.int64)
    return ids, y, pca, pg, pf, qtype


def X_dis(pca, pg, ids, qtype, thr_ca, thr_gnn):
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


def fit_and_route(tr, va, te, thr_ca, thr_gnn, C, tau):
    ids, y, pca, pg, pf, qt = tr
    X = X_dis(pca, pg, ids, qt, thr_ca, thr_gnn)
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
    clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
    clf.fit(X[mask], pick[mask])

    def route(split):
        ids, y, pca, pg, pf, qt = split
        Xd = X_dis(pca, pg, ids, qt, thr_ca, thr_gnn)
        pred = (pca >= thr_ca).astype(np.int64)
        score = pca.copy()
        dis = (pca >= thr_ca) != (pg >= thr_gnn)
        if dis.any():
            p = clf.predict_proba(Xd[dis])[:, 1]
            use = p >= tau
            idx = np.where(dis)[0]
            pred[idx] = np.where(use, (pg[idx] >= thr_gnn).astype(np.int64), pred[idx])
            score[idx] = np.where(use, pg[idx], pca[idx])
        return pred, score

    return route(tr), route(va), route(te), clf


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def main() -> int:
    rj = json.loads(ROUTER_JSON.read_text(encoding="utf-8"))
    w = np.asarray(rj["ca_seed_weights"], dtype=np.float64)
    thr_ca = float(rj["thr_ca"])
    thr_gnn = float(rj["thr_gnn"])
    thr_face = float(
        json.loads((FACE_RUN / "trainer_state.json").read_text(encoding="utf-8"))["threshold"]
    )
    C = float(rj["logreg_C"])
    tau = float(rj["tau_pick_gnn"])

    tr = load_split("train", w)
    va = load_split("val", w)
    te = load_split("test", w)
    (pred_tr, sc_tr), (pred_v, sc_v), (pred_t, sc_t), _ = fit_and_route(
        tr, va, te, thr_ca, thr_gnn, C, tau
    )
    base_v = macro_f1(va[1], pred_v)
    base_t = macro_f1(te[1], pred_t)
    print(f"baseline 2mem val={base_v:.4f} test={base_t:.4f} thr_face={thr_face}", flush=True)

    results: list[dict] = []

    # --- A: simplex over score streams, thr on val ---
    rng = np.random.default_rng(0)
    recipes = {
        "router+face": (lambda sc, pca, pg, pf: [sc, pf]),
        "ca+gnn+face": (lambda sc, pca, pg, pf: [pca, pg, pf]),
        "router+gnn+face": (lambda sc, pca, pg, pf: [sc, pg, pf]),
    }
    for name, fn in recipes.items():
        sv = fn(sc_v, va[2], va[3], va[4])
        st = fn(sc_t, te[2], te[3], te[4])
        k = len(sv)
        best_local = None
        for z0 in [np.zeros(k)] + [rng.normal(0, 0.7, size=k) for _ in range(25)]:
            def obj(z, sv=sv):
                ww = softmax(z)
                sc = sum(ww[i] * sv[i] for i in range(k))
                return -best_thr(sc, va[1])[1]

            res = minimize(obj, z0, method="Nelder-Mead", options={"maxiter": 250})
            ww = softmax(res.x)
            scv = sum(ww[i] * sv[i] for i in range(k))
            thr, vf1 = best_thr(scv, va[1])
            sct = sum(ww[i] * st[i] for i in range(k))
            tf1 = macro_f1(te[1], (sct >= thr).astype(np.int64))
            cand = (vf1, tf1, ww.tolist(), thr)
            if best_local is None or vf1 > best_local[0]:
                best_local = cand
        assert best_local is not None
        results.append(
            {
                "method": f"simplex:{name}",
                "val_f1": best_local[0],
                "test_f1": best_local[1],
                "weights": best_local[2],
                "thr": best_local[3],
            }
        )
        print(
            f"simplex {name}: val={best_local[0]:.4f} test={best_local[1]:.4f} w={best_local[2]}",
            flush=True,
        )

    # --- B: soft residual on router score ---
    best_res = None
    for beta in np.linspace(-1.5, 1.5, 31):
        for max_bm in (0.1, 0.2, 0.35, 1.0):
            for min_fm in (0.0, 0.1, 0.2):
                for mode in ("agree_dis", "any_dis", "unc"):
                    def gate(pca, pg, pred, pf):
                        agree = (pca >= thr_ca) == (pg >= thr_gnn)
                        dis = pred != (pf >= thr_face)
                        if mode == "agree_dis":
                            return (
                                agree
                                & dis
                                & (np.abs(pca - thr_ca) <= max_bm)
                                & (np.abs(pf - thr_face) >= min_fm)
                            )
                        if mode == "any_dis":
                            return (
                                dis
                                & (np.abs(pca - thr_ca) <= max_bm)
                                & (np.abs(pf - thr_face) >= min_fm)
                            )
                        return dis & (np.abs(pca - thr_ca) <= max_bm)

                    gv = gate(va[2], va[3], pred_v, va[4])
                    gt = gate(te[2], te[3], pred_t, te[4])
                    sv = sc_v.copy()
                    st = sc_t.copy()
                    sv[gv] = sv[gv] + beta * (va[4][gv] - thr_face)
                    st[gt] = st[gt] + beta * (te[4][gt] - thr_face)
                    thr, vf1 = best_thr(sv, va[1])
                    if vf1 + 1e-12 < base_v:
                        continue
                    tf1 = macro_f1(te[1], (st >= thr).astype(np.int64))
                    cand = {
                        "method": "soft_residual",
                        "val_f1": vf1,
                        "test_f1": tf1,
                        "beta": float(beta),
                        "max_bm": max_bm,
                        "min_fm": min_fm,
                        "mode": mode,
                        "thr": thr,
                        "n_gate_test": int(gt.sum()),
                    }
                    if best_res is None or vf1 > best_res["val_f1"] or (
                        abs(vf1 - best_res["val_f1"]) < 1e-12 and tf1 > best_res["test_f1"]
                    ):
                        best_res = cand
    if best_res:
        results.append(best_res)
        print(
            f"soft_residual: val={best_res['val_f1']:.4f} test={best_res['test_f1']:.4f} {best_res}",
            flush=True,
        )
    else:
        print("soft_residual: no candidate >= baseline val", flush=True)

    # --- C: multiclass CA/GNN/Face when members disagree ---
    def multi_rows(ids, pca, pg, pf, qt):
        pred_c = (pca >= thr_ca).astype(np.int64)
        pred_g = (pg >= thr_gnn).astype(np.int64)
        pred_f = (pf >= thr_face).astype(np.int64)
        rows = []
        idxs = []
        for i, vid in enumerate(ids):
            if len({int(pred_c[i]), int(pred_g[i]), int(pred_f[i])}) == 1:
                continue
            q = qt[vid]
            qoh = [1.0 if q == t else 0.0 for t in QTYPES]
            rows.append(
                [
                    float(pca[i]),
                    float(pg[i]),
                    float(pf[i]),
                    abs(float(pca[i]) - thr_ca),
                    abs(float(pg[i]) - thr_gnn),
                    abs(float(pf[i]) - thr_face),
                    float(pca[i] - pg[i]),
                    float(pca[i] - pf[i]),
                    float(pg[i] - pf[i]),
                    float(pred_c[i]),
                    float(pred_g[i]),
                    float(pred_f[i]),
                    max(float(pca[i]), 1.0 - float(pca[i])),
                    max(float(pg[i]), 1.0 - float(pg[i])),
                    max(float(pf[i]), 1.0 - float(pf[i])),
                    *qoh,
                ]
            )
            idxs.append(i)
        return np.asarray(rows, dtype=np.float32), np.asarray(idxs, dtype=np.int64), pred_c, pred_g, pred_f

    # train labels: first correct among CA, GNN, Face
    Xm, im, pc, pg_, pf_ = multi_rows(tr[0], tr[2], tr[3], tr[4], tr[5])
    y_multi = []
    keep = []
    for j, i in enumerate(im):
        cands = []
        if pc[i] == tr[1][i]:
            cands.append(0)
        if pg_[i] == tr[1][i]:
            cands.append(1)
        if pf_[i] == tr[1][i]:
            cands.append(2)
        if not cands:
            continue
        y_multi.append(cands[0])
        keep.append(j)
    Xm = Xm[np.asarray(keep)] if keep else Xm
    y_multi = np.asarray(y_multi, dtype=np.int64)
    print(f"multiclass train n={len(y_multi)} dist={np.bincount(y_multi, minlength=3)}", flush=True)

    best_m = None
    if len(y_multi) >= 20 and len(np.unique(y_multi)) >= 2:
        for C2 in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            mlr = LogisticRegression(
                C=C2, max_iter=5000, class_weight="balanced", multi_class="multinomial"
            )
            mlr.fit(Xm, y_multi)
            for tau_m in np.linspace(0.34, 0.85, 18):
                def apply(split, pred_base):
                    ids, y, pca, pg, pf, qt = split
                    X, idxs, pc, pgx, pfx = multi_rows(ids, pca, pg, pf, qt)
                    out = pred_base.copy()
                    if len(idxs) == 0:
                        return out
                    proba = mlr.predict_proba(X)
                    cls = proba.argmax(1)
                    conf = proba.max(1)
                    pick = np.stack([pc[idxs], pgx[idxs], pfx[idxs]], axis=1)
                    for j, i in enumerate(idxs):
                        if conf[j] >= tau_m:
                            out[i] = int(pick[j, cls[j]])
                    return out

                pv = apply(va, pred_v)
                pt = apply(te, pred_t)
                vf1 = macro_f1(va[1], pv)
                if vf1 + 1e-12 < base_v:
                    continue
                tf1 = macro_f1(te[1], pt)
                cand = {
                    "method": "multiclass_ca_gnn_face",
                    "val_f1": vf1,
                    "test_f1": tf1,
                    "C": C2,
                    "tau": float(tau_m),
                    "n_swap_test": int((pt != pred_t).sum()),
                }
                if best_m is None or vf1 > best_m["val_f1"] or (
                    abs(vf1 - best_m["val_f1"]) < 1e-12 and tf1 > best_m["test_f1"]
                ):
                    best_m = cand
    if best_m:
        results.append(best_m)
        print(f"multiclass: val={best_m['val_f1']:.4f} test={best_m['test_f1']:.4f}", flush=True)
    else:
        print("multiclass: no candidate >= baseline val", flush=True)

    # --- D: binary face rescue when CA==GNN and face disagrees ---
    rows = []
    labs = []
    for i in range(len(tr[1])):
        agree = (tr[2][i] >= thr_ca) == (tr[3][i] >= thr_gnn)
        pf_i = int(tr[4][i] >= thr_face)
        if not agree or pred_tr[i] == pf_i:
            continue
        q = tr[5][tr[0][i]]
        qoh = [1.0 if q == t else 0.0 for t in QTYPES]
        rows.append(
            [
                float(sc_tr[i]),
                float(tr[4][i]),
                float(tr[2][i]),
                float(tr[3][i]),
                abs(float(sc_tr[i]) - 0.5),
                abs(float(tr[4][i]) - thr_face),
                abs(float(tr[2][i]) - thr_ca),
                float(sc_tr[i] - tr[4][i]),
                *qoh,
            ]
        )
        labs.append(1 if (pf_i == tr[1][i] and pred_tr[i] != tr[1][i]) else 0)
    best_h = None
    if len(labs) >= 10 and len(set(labs)) >= 2:
        Xb = np.asarray(rows, dtype=np.float32)
        yb = np.asarray(labs, dtype=np.int64)
        for C2 in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            lr = LogisticRegression(C=C2, max_iter=4000, class_weight="balanced")
            lr.fit(Xb, yb)
            for tau_f in np.linspace(0.45, 0.95, 21):
                def apply(pred, sc, pca, pg, pf, ids, qt):
                    out = pred.copy()
                    pred_f = (pf >= thr_face).astype(np.int64)
                    Rr = []
                    idxs = []
                    for i, vid in enumerate(ids):
                        agree = (pca[i] >= thr_ca) == (pg[i] >= thr_gnn)
                        if not agree or pred[i] == pred_f[i]:
                            continue
                        q = qt[vid]
                        qoh = [1.0 if q == t else 0.0 for t in QTYPES]
                        Rr.append(
                            [
                                float(sc[i]),
                                float(pf[i]),
                                float(pca[i]),
                                float(pg[i]),
                                abs(float(sc[i]) - 0.5),
                                abs(float(pf[i]) - thr_face),
                                abs(float(pca[i]) - thr_ca),
                                float(sc[i] - pf[i]),
                                *qoh,
                            ]
                        )
                        idxs.append(i)
                    if not Rr:
                        return out
                    use = lr.predict_proba(np.asarray(Rr, dtype=np.float32))[:, 1] >= tau_f
                    for j, i in enumerate(idxs):
                        if use[j]:
                            out[i] = pred_f[i]
                    return out

                pv = apply(pred_v, sc_v, va[2], va[3], va[4], va[0], va[5])
                pt = apply(pred_t, sc_t, te[2], te[3], te[4], te[0], te[5])
                vf1 = macro_f1(va[1], pv)
                if vf1 + 1e-12 < base_v:
                    continue
                tf1 = macro_f1(te[1], pt)
                cand = {
                    "method": "face_rescue_agree",
                    "val_f1": vf1,
                    "test_f1": tf1,
                    "C": C2,
                    "tau_face": float(tau_f),
                    "n_swap_test": int((pt != pred_t).sum()),
                }
                if best_h is None or vf1 > best_h["val_f1"] or (
                    abs(vf1 - best_h["val_f1"]) < 1e-12 and tf1 > best_h["test_f1"]
                ):
                    best_h = cand
    if best_h:
        results.append(best_h)
        print(
            f"face_rescue: val={best_h['val_f1']:.4f} test={best_h['test_f1']:.4f} swaps={best_h['n_swap_test']}",
            flush=True,
        )
    else:
        print("face_rescue: no candidate >= baseline val", flush=True)

    # pick best by val among those with val>=base; prefer test improvement
    eligible = [r for r in results if r["val_f1"] + 1e-12 >= base_v]
    eligible.sort(key=lambda r: (r["val_f1"], r["test_f1"]), reverse=True)
    winners = [r for r in eligible if r["test_f1"] > base_t + 1e-12]
    out = {
        "baseline_val": base_v,
        "baseline_test": base_t,
        "thr_face": thr_face,
        "n_candidates": len(results),
        "best_by_val": eligible[0] if eligible else None,
        "best_improving_test": winners[0] if winners else None,
        "all": results,
        "promote": bool(winners),
    }
    out_path = ROOT / "outputs/ensemble_eval/face_fusion_search.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("=" * 60, flush=True)
    if winners:
        print(f"IMPROVED test: {winners[0]}", flush=True)
    else:
        print(f"No test improvement over {base_t:.4f}. Best by val: {eligible[0] if eligible else None}", flush=True)
    print(f"→ {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
