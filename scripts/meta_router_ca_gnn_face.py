#!/usr/bin/env python3
"""Meta-router CA ⊕ GNN ⊕ Face — 3º membro com rescue em CA==GNN.

Pipeline (sem peek no test; sem features de anotação):

1. Pesos dos 7 seeds CA (candidatos no train).
2. LogReg discordância CA↔GNN (como meta_router_ca_gnn).
3. LogReg rescue: quando CA==GNN e face discorda → P(usar face).
4. Seleção conjunta na val (w, C, τ, C_face, τ_face); test 1×.

Uso::

    uv run python scripts/meta_router_ca_gnn_face.py \\
      --face-run outputs/face_gnn_ts/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from src.training.metrics import evaluate_video_predictions

from _runs import ca_seeds, run_dir  # noqa: E402 — overrides BAH_* (scripts/_runs.py)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ca_seeds([
    "20260713_175242",
    "20260713_175257",
    "20260713_175312",
    "20260713_175327",
    "20260713_175342",
    "20260713_175357",
    "20260713_175412",
])
GNN_RUN = run_dir("BAH_GNN_RUN", "outputs/hetero_gnn_contrastive/20260713_162753")
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


def _load_proba(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        return {r["video_id"]: float(r["y_proba"]) for r in csv.DictReader(f)}


def _load_split(split: str, face_run: Path):
    mats: list[dict[str, float]] = []
    labels: dict[str, int] = {}
    qtype: dict[str, str] = {}
    for i, seed in enumerate(SEEDS):
        path = ROOT / "outputs" / "cross_attention" / seed / f"eval_{split}" / "predictions.csv"
        d: dict[str, float] = {}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d[row["video_id"]] = float(row["y_proba"])
                labels[row["video_id"]] = int(row["y_true"])
                if i == 0:
                    qtype[row["video_id"]] = str(row.get("question_type") or "?")
        mats.append(d)
    ids = sorted(labels)
    ca = np.array([[mats[j][vid] for j in range(7)] for vid in ids], dtype=np.float32)

    gnn = _load_proba(ROOT / GNN_RUN / f"eval_{split}" / "predictions.csv")
    face = _load_proba(ROOT / face_run / f"eval_{split}" / "predictions.csv")
    missing = [vid for vid in ids if vid not in gnn or vid not in face]
    if missing:
        raise KeyError(f"{split}: {len(missing)} ids sem GNN/face (ex: {missing[:3]})")
    pg = np.array([gnn[vid] for vid in ids], dtype=np.float32)
    pf = np.array([face[vid] for vid in ids], dtype=np.float32)
    y = np.array([labels[vid] for vid in ids], dtype=np.int64)
    return ids, y, ca, pg, pf, qtype


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


def _features_disagree(
    pca: np.ndarray,
    pg: np.ndarray,
    pf: np.ndarray,
    ids: list[str],
    qtype: dict[str, str],
    thr_ca: float,
    thr_gnn: float,
    thr_face: float,
) -> np.ndarray:
    rows = []
    for i, vid in enumerate(ids):
        q = qtype.get(vid, "?")
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
                abs(float(pca[i] - pg[i])),
                float(pca[i] - pf[i]),
                float(pg[i] - pf[i]),
                float((pca[i] >= thr_ca) == (pg[i] >= thr_gnn)),
                float((pca[i] >= thr_ca) == (pf[i] >= thr_face)),
                float(pca[i] >= thr_ca),
                float(pg[i] >= thr_gnn),
                float(pf[i] >= thr_face),
                max(float(pca[i]), 1.0 - float(pca[i])),
                max(float(pg[i]), 1.0 - float(pg[i])),
                max(float(pf[i]), 1.0 - float(pf[i])),
                *qoh,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _pick_gnn_labels(
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


def _pick_face_rescue_labels(
    pca: np.ndarray,
    pg: np.ndarray,
    pf: np.ndarray,
    y: np.ndarray,
    thr_ca: float,
    thr_gnn: float,
    thr_face: float,
    max_base_margin: float = 0.12,
    min_face_margin: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """CA==GNN, face discorda, base inseguro, face confiante → 1 se face acerta e base erra."""
    pred_ca = pca >= thr_ca
    pred_g = pg >= thr_gnn
    pred_f = pf >= thr_face
    agree = pred_ca == pred_g
    disagree_f = pred_f != pred_ca
    base_margin = np.abs(pca - thr_ca)
    face_margin = np.abs(pf - thr_face)
    mask = (
        agree
        & disagree_f
        & (base_margin <= max_base_margin)
        & (face_margin >= min_face_margin)
    )
    base_ok = pred_ca == y
    face_ok = pred_f == y
    pick = np.zeros(len(y), dtype=np.int64)
    pick[mask] = ((~base_ok) & face_ok)[mask].astype(np.int64)
    return mask, pick


def _route(
    clf_g: LogisticRegression,
    clf_f: LogisticRegression | None,
    X: np.ndarray,
    pca: np.ndarray,
    pg: np.ndarray,
    pf: np.ndarray,
    thr_ca: float,
    thr_gnn: float,
    thr_face: float,
    tau: float,
    tau_face: float,
    max_base_margin: float = 0.12,
    min_face_margin: float = 0.15,
) -> np.ndarray:
    pred_ca = (pca >= thr_ca).astype(np.int64)
    pred_g = (pg >= thr_gnn).astype(np.int64)
    pred_f = (pf >= thr_face).astype(np.int64)
    pred = pred_ca.copy()

    disagree = pred_ca != pred_g
    if disagree.any():
        p_gnn = clf_g.predict_proba(X[disagree])[:, 1]
        use_g = p_gnn >= tau
        idx = np.where(disagree)[0]
        pred[idx] = np.where(use_g, pred_g[idx], pred_ca[idx])

    if clf_f is not None:
        agree = pred_ca == pred_g
        base_margin = np.abs(pca - thr_ca)
        face_margin = np.abs(pf - thr_face)
        rescue_mask = (
            agree
            & (pred_f != pred_ca)
            & (base_margin <= max_base_margin)
            & (face_margin >= min_face_margin)
        )
        if rescue_mask.any():
            p_face = clf_f.predict_proba(X[rescue_mask])[:, 1]
            use_f = p_face >= tau_face
            idx = np.where(rescue_mask)[0]
            pred[idx] = np.where(use_f, pred_f[idx], pred[idx])
    return pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-run", type=Path, required=True)
    parser.add_argument("--thr-gnn", type=float, default=None)
    parser.add_argument("--thr-face", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/ensemble_eval/meta_router_ca_gnn_face.json",
    )
    args = parser.parse_args()

    face_run = args.face_run
    ids_tr, y_tr, ca_tr, pg_tr, pf_tr, qt_tr = _load_split("train", face_run)
    ids_v, y_v, ca_v, pg_v, pf_v, qt_v = _load_split("val", face_run)
    ids_t, y_t, ca_t, pg_t, pf_t, qt_t = _load_split("test", face_run)

    thr_gnn = args.thr_gnn
    if thr_gnn is None:
        thr_gnn = float(
            json.loads((ROOT / GNN_RUN / "trainer_state.json").read_text(encoding="utf-8"))[
                "threshold"
            ]
        )
    thr_face = args.thr_face
    if thr_face is None:
        thr_face = float(
            json.loads((ROOT / face_run / "trainer_state.json").read_text(encoding="utf-8"))[
                "threshold"
            ]
        )

    weight_cands = _candidate_weights(ca_tr, y_tr, seed=args.seed)
    print(f"weight candidates: {len(weight_cands)}")

    # Prefer locked produção 2-membro se existir (evita regressão ao retunar C/τ junto com face).
    locked = ROOT / "outputs/ensemble_eval/meta_router_ca_gnn.json"
    if locked.exists():
        lj = json.loads(locked.read_text(encoding="utf-8"))
        w0 = np.asarray(lj["ca_seed_weights"], dtype=np.float64)
        weight_cands = [w0] + [w for w in weight_cands if not np.allclose(w, w0, atol=1e-3)]

    best: dict | None = None
    for w in weight_cands[:8]:  # poucos pesos; foco no rescue face
        pca_tr = (ca_tr * w).sum(1)
        pca_v = (ca_v * w).sum(1)
        if locked.exists() and np.allclose(w, w0, atol=1e-3):
            thr_ca = float(lj["thr_ca"])
            C_list = [float(lj["logreg_C"])]
            tau_list = [float(lj["tau_pick_gnn"])]
        else:
            thr_ca, _ = _best_thr(pca_v, y_v)
            C_list = [0.5, 1.0, 2.0]
            tau_list = list(np.linspace(0.35, 0.55, 5))
        X_tr = _features_disagree(
            pca_tr, pg_tr, pf_tr, ids_tr, qt_tr, thr_ca, thr_gnn, thr_face
        )
        X_v = _features_disagree(
            pca_v, pg_v, pf_v, ids_v, qt_v, thr_ca, thr_gnn, thr_face
        )
        mask_g, pick_g = _pick_gnn_labels(pca_tr, pg_tr, y_tr, thr_ca, thr_gnn)
        if int(mask_g.sum()) < 10 or len(np.unique(pick_g[mask_g])) < 2:
            continue
        ca_alone_v = _macro_f1(y_v, (pca_v >= thr_ca).astype(np.int64))
        for C in C_list:
            clf_g = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
            clf_g.fit(X_tr[mask_g], pick_g[mask_g])
            for tau in tau_list:
                # baseline 2-membro nesta (w,C,τ)
                pred_v2 = _route(
                    clf_g,
                    None,
                    X_v,
                    pca_v,
                    pg_v,
                    pf_v,
                    thr_ca,
                    thr_gnn,
                    thr_face,
                    float(tau),
                    1.1,
                )
                f1_2 = _macro_f1(y_v, pred_v2)
                # também registra baseline como candidato
                cand2 = {
                    "val_f1": f1_2,
                    "n_swap_val": int((pred_v2 != (pca_v >= thr_ca)).sum()),
                    "C": float(C),
                    "Cf": None,
                    "tau": float(tau),
                    "tau_face": 1.1,
                    "max_base_margin": 0.0,
                    "min_face_margin": 1.0,
                    "thr_ca": float(thr_ca),
                    "w": w,
                    "clf_g": clf_g,
                    "clf_f": None,
                    "ca_alone_v": ca_alone_v,
                }
                if best is None or cand2["val_f1"] > best["val_f1"]:
                    best = cand2

                for max_bm in [0.08, 0.12, 0.18, 0.25]:
                    for min_fm in [0.10, 0.15, 0.20, 0.25]:
                        mask_f, pick_f = _pick_face_rescue_labels(
                            pca_tr,
                            pg_tr,
                            pf_tr,
                            y_tr,
                            thr_ca,
                            thr_gnn,
                            thr_face,
                            max_base_margin=max_bm,
                            min_face_margin=min_fm,
                        )
                        if int(mask_f.sum()) < 6 or len(np.unique(pick_f[mask_f])) < 2:
                            continue
                        for Cf in [0.1, 0.5, 1.0, 2.0]:
                            clf_f = LogisticRegression(
                                C=float(Cf), max_iter=4000, class_weight="balanced"
                            )
                            clf_f.fit(X_tr[mask_f], pick_f[mask_f])
                            for tau_f in np.linspace(0.45, 0.80, 8):
                                pred_v = _route(
                                    clf_g,
                                    clf_f,
                                    X_v,
                                    pca_v,
                                    pg_v,
                                    pf_v,
                                    thr_ca,
                                    thr_gnn,
                                    thr_face,
                                    float(tau),
                                    float(tau_f),
                                    max_base_margin=max_bm,
                                    min_face_margin=min_fm,
                                )
                                val_f1 = _macro_f1(y_v, pred_v)
                                # só aceita face se NÃO piorar o 2-membro na val
                                if val_f1 + 1e-12 < f1_2:
                                    continue
                                n_swap = int((pred_v != (pca_v >= thr_ca)).sum())
                                cand = {
                                    "val_f1": val_f1,
                                    "n_swap_val": n_swap,
                                    "C": float(C),
                                    "Cf": float(Cf),
                                    "tau": float(tau),
                                    "tau_face": float(tau_f),
                                    "max_base_margin": float(max_bm),
                                    "min_face_margin": float(min_fm),
                                    "thr_ca": float(thr_ca),
                                    "w": w,
                                    "clf_g": clf_g,
                                    "clf_f": clf_f,
                                    "ca_alone_v": ca_alone_v,
                                }
                                if best is None or val_f1 > best["val_f1"] or (
                                    abs(val_f1 - best["val_f1"]) < 1e-12
                                    and n_swap > best["n_swap_val"]
                                    and cand["Cf"] is not None
                                ):
                                    best = cand

    if best is None:
        raise RuntimeError("Nenhum candidato válido para o roteador")

    w = best["w"]
    thr_ca = best["thr_ca"]
    clf_g = best["clf_g"]
    clf_f = best["clf_f"]
    tau = best["tau"]
    tau_face = best["tau_face"]
    max_bm = float(best["max_base_margin"])
    min_fm = float(best["min_face_margin"])

    pca_t = (ca_t * w).sum(1)
    X_t = _features_disagree(pca_t, pg_t, pf_t, ids_t, qt_t, thr_ca, thr_gnn, thr_face)
    pred_t = _route(
        clf_g,
        clf_f,
        X_t,
        pca_t,
        pg_t,
        pf_t,
        thr_ca,
        thr_gnn,
        thr_face,
        tau,
        tau_face,
        max_base_margin=max_bm,
        min_face_margin=min_fm,
    )

    pred_ca_t = (pca_t >= thr_ca).astype(np.int64)
    pred_g_t = (pg_t >= thr_gnn).astype(np.int64)
    pred_f_t = (pf_t >= thr_face).astype(np.int64)
    n_swap_gnn = int(
        ((pred_ca_t != pred_g_t) & (pred_t == pred_g_t) & (pred_t != pred_ca_t)).sum()
    )
    n_swap_face = int(
        ((pred_t == pred_f_t) & (pred_t != pred_ca_t) & (pred_ca_t == pred_g_t)).sum()
    )

    scores_arr = np.where(pred_t == pred_ca_t, pca_t, pca_t)
    scores_arr = np.where(
        (pred_t != pred_ca_t) & (pred_t == pred_g_t),
        pg_t,
        scores_arr,
    )
    scores_arr = np.where(
        (pred_t != pred_ca_t) & (pred_t == pred_f_t),
        pf_t,
        scores_arr,
    )

    labels = {vid: int(y) for vid, y in zip(ids_t, y_t, strict=True)}
    preds = {vid: int(p) for vid, p in zip(ids_t, pred_t, strict=True)}
    scores = {vid: float(s) for vid, s in zip(ids_t, scores_arr, strict=True)}
    report = evaluate_video_predictions(labels, preds, scores)

    # Baseline 2-member (same GNN clf, no face rescue)
    pred_2 = pred_ca_t.copy()
    disagree = pred_ca_t != pred_g_t
    if disagree.any():
        p_gnn = clf_g.predict_proba(X_t[disagree])[:, 1]
        use_g = p_gnn >= tau
        idx = np.where(disagree)[0]
        pred_2[idx] = np.where(use_g, pred_g_t[idx], pred_ca_t[idx])
    f1_2 = _macro_f1(y_t, pred_2)

    out = {
        "method": "locked_2member_plus_gated_face_rescue",
        "face_run": str(face_run),
        "ca_seed_weights": w.tolist(),
        "thr_ca": thr_ca,
        "thr_gnn": thr_gnn,
        "thr_face": thr_face,
        "logreg_C": best["C"],
        "logreg_C_face": best["Cf"],
        "tau_pick_gnn": tau,
        "tau_pick_face": tau_face,
        "max_base_margin": max_bm,
        "min_face_margin": min_fm,
        "val_f1_router": best["val_f1"],
        "val_f1_ca_alone": best["ca_alone_v"],
        "n_swap_val": best["n_swap_val"],
        "n_swap_gnn_test": n_swap_gnn,
        "n_swap_face_test": n_swap_face,
        "test_f1": report["macro_f1"],
        "test_f1_2member_same_gnn_clf": f1_2,
        "test_ca_alone": _macro_f1(y_t, pred_ca_t),
        "test_gnn_alone": _macro_f1(y_t, pred_g_t),
        "test_face_alone": _macro_f1(y_t, pred_f_t),
        "report": report,
        "promote": bool(report["macro_f1"] > 0.7454),
        "note": (
            "Face rescue só se val ≥ 2-membro; gates de margem base/face. "
            "Promove produção só se test_f1 > 0.7454."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary = {k: out[k] for k in out if k != "report"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"TEST macro_f1={report['macro_f1']:.4f} (2-member={f1_2:.4f}) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
