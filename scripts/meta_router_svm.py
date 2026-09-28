#!/usr/bin/env python3
"""Meta-router CA⊕GNN via SVM 2D (scores CA × GNN).

Treina no train, seleciona kernel/C/γ/thr na val, avalia 1× no test.
Usa os pesos CA travados do meta-router LogReg (produção).

Uso::

    uv run python scripts/meta_router_svm.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
ROUTER_JSON = ROOT / "outputs/ensemble_eval/meta_router_ca_gnn.json"
FIG_DIR = ROOT / "references/figures/meta_router_ca_gnn"


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
    best_f1, best_thr = -1.0, 0.0
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return 0.0, _macro_f1(y, np.ones_like(y) * int(scores.mean() >= 0))
    for thr in np.linspace(lo, hi, 201):
        f1 = _macro_f1(y, (scores >= thr).astype(np.int64))
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def _load_split(split: str, w: np.ndarray):
    mats: list[dict[str, float]] = []
    labels: dict[str, int] = {}
    for seed in SEEDS:
        path = ROOT / "outputs" / "cross_attention" / seed / f"eval_{split}" / "predictions.csv"
        d: dict[str, float] = {}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d[row["video_id"]] = float(row["y_proba"])
                labels[row["video_id"]] = int(row["y_true"])
        mats.append(d)
    ids = sorted(labels)
    ca = np.array([[mats[j][vid] for j in range(7)] for vid in ids], dtype=np.float64)
    pca = (ca * w).sum(1)
    with (ROOT / GNN_RUN / f"eval_{split}" / "predictions.csv").open(encoding="utf-8") as f:
        gnn = {r["video_id"]: float(r["y_proba"]) for r in csv.DictReader(f)}
    pg = np.array([gnn[vid] for vid in ids], dtype=np.float64)
    y = np.array([labels[vid] for vid in ids], dtype=np.int64)
    return ids, y, pca, pg


def _feats(pca: np.ndarray, pg: np.ndarray) -> np.ndarray:
    """2D base + features polinomiais leves (úteis p/ linear)."""
    return np.column_stack(
        [
            pca,
            pg,
            pca - pg,
            np.abs(pca - pg),
            pca * pg,
            pca**2,
            pg**2,
        ]
    ).astype(np.float64)


def _param_grid() -> list[dict]:
    grid: list[dict] = []
    for C in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        grid.append({"kernel": "linear", "C": C, "gamma": "scale", "degree": 3})
    for C in (0.5, 1.0, 2.0, 5.0, 10.0):
        for gamma in ("scale", "auto", 0.5, 1.0, 2.0, 5.0):
            grid.append({"kernel": "rbf", "C": C, "gamma": gamma, "degree": 3})
    for C in (0.5, 1.0, 2.0, 5.0):
        for degree in (2, 3):
            for gamma in ("scale", 1.0, 2.0):
                grid.append({"kernel": "poly", "C": C, "gamma": gamma, "degree": degree})
    for C in (0.5, 1.0, 2.0, 5.0):
        for gamma in ("scale", 1.0, 2.0):
            grid.append({"kernel": "sigmoid", "C": C, "gamma": gamma, "degree": 3})
    return grid


def _fit_svm(X: np.ndarray, y: np.ndarray, params: dict) -> Pipeline:
    clf = SVC(
        kernel=params["kernel"],
        C=params["C"],
        gamma=params["gamma"],
        degree=int(params["degree"]),
        class_weight="balanced",
        cache_size=512,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", clf)])
    pipe.fit(X, y)
    return pipe


def _decision(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipe.decision_function(X)


def _plot_boundary(
    pipe: Pipeline,
    thr: float,
    pca: np.ndarray,
    pg: np.ndarray,
    y: np.ndarray,
    title: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    m0, m1 = y == 0, y == 1
    ax.scatter(pg[m0], pca[m0], c="#16a34a", s=22, alpha=0.7, edgecolors="none", label=f"sem A/H n={m0.sum()}")
    ax.scatter(pg[m1], pca[m1], c="#dc2626", s=22, alpha=0.7, edgecolors="none", label=f"com A/H n={m1.sum()}")

    xx = np.linspace(0, 1, 220)
    yy = np.linspace(0, 1, 220)
    XX, YY = np.meshgrid(xx, yy)
    grid = _feats(YY.ravel(), XX.ravel())  # y=CA, x=GNN
    Z = (_decision(pipe, grid) >= thr).astype(np.float64).reshape(XX.shape)
    ax.contour(XX, YY, Z, levels=[0.5], colors=["#111827"], linewidths=1.8)
    ax.contourf(XX, YY, Z, levels=[-0.1, 0.5, 1.1], colors=["#bbf7d0", "#fecaca"], alpha=0.18)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("score GNN  P(A/H)")
    ax.set_ylabel("score CA (7-seed weighted)  P(A/H)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/ensemble_eval/meta_router_svm.json",
    )
    args = parser.parse_args()

    rj = json.loads(ROUTER_JSON.read_text(encoding="utf-8"))
    w = np.asarray(rj["ca_seed_weights"], dtype=np.float64)
    baseline = float(rj["test_f1"])

    ids_tr, y_tr, pca_tr, pg_tr = _load_split("train", w)
    ids_v, y_v, pca_v, pg_v = _load_split("val", w)
    ids_t, y_t, pca_t, pg_t = _load_split("test", w)

    X_tr = _feats(pca_tr, pg_tr)
    X_v = _feats(pca_v, pg_v)
    X_t = _feats(pca_t, pg_t)

    # baselines 1D
    ca_alone = _macro_f1(y_t, (pca_t >= float(rj["thr_ca"])).astype(np.int64))
    gnn_alone = _macro_f1(y_t, (pg_t >= float(rj["thr_gnn"])).astype(np.int64))

    best: dict | None = None
    rows: list[dict] = []
    for params in _param_grid():
        try:
            pipe = _fit_svm(X_tr, y_tr, params)
            s_v = _decision(pipe, X_v)
            thr, val_f1 = _best_thr(s_v, y_v)
            pred_v = (s_v >= thr).astype(np.int64)
            # also record default predict for reference
            pred_default = pipe.predict(X_v).astype(np.int64)
            val_default = _macro_f1(y_v, pred_default)
            cand = {
                **params,
                "thr": thr,
                "val_f1": val_f1,
                "val_f1_default_predict": val_default,
                "pipe": pipe,
            }
            rows.append({k: cand[k] for k in cand if k != "pipe"})
            if best is None or val_f1 > best["val_f1"] or (
                abs(val_f1 - best["val_f1"]) < 1e-12
                and val_default > best.get("val_f1_default_predict", -1)
            ):
                best = cand
        except Exception as exc:  # noqa: BLE001
            rows.append({**params, "error": str(exc)})

    if best is None:
        raise RuntimeError("Nenhum SVM válido")

    pipe = best["pipe"]
    thr = float(best["thr"])
    s_t = _decision(pipe, X_t)
    pred_t = (s_t >= thr).astype(np.int64)
    test_f1 = _macro_f1(y_t, pred_t)

    # scores for metrics (map decision to [0,1]-ish via sigmoid of shifted margin)
    # keep raw decision as score for AP/ROC
    labels = {vid: int(y) for vid, y in zip(ids_t, y_t, strict=True)}
    preds = {vid: int(p) for vid, p in zip(ids_t, pred_t, strict=True)}
    scores = {vid: float(s) for vid, s in zip(ids_t, s_t, strict=True)}
    report = evaluate_video_predictions(labels, preds, scores)

    # top-5 by val
    ok_rows = [r for r in rows if "val_f1" in r]
    ok_rows.sort(key=lambda r: r["val_f1"], reverse=True)
    top5 = ok_rows[:5]

    # also evaluate top kernels on test for transparency (still selected by val first)
    per_kernel_best: dict[str, dict] = {}
    for r in ok_rows:
        k = r["kernel"]
        if k not in per_kernel_best or r["val_f1"] > per_kernel_best[k]["val_f1"]:
            per_kernel_best[k] = r

    # re-eval each kernel's best-val config on test
    kernel_test = {}
    for k, r in per_kernel_best.items():
        p = _fit_svm(X_tr, y_tr, r)
        thr_k, _ = _best_thr(_decision(p, X_v), y_v)
        pred = (_decision(p, X_t) >= thr_k).astype(np.int64)
        kernel_test[k] = {
            "val_f1": r["val_f1"],
            "test_f1": _macro_f1(y_t, pred),
            "C": r["C"],
            "gamma": r["gamma"],
            "degree": r.get("degree"),
            "thr": thr_k,
        }

    title = (
        f"SVM {best['kernel']} C={best['C']} γ={best['gamma']} · "
        f"val={best['val_f1']:.3f} test={test_f1:.3f}"
    )
    _plot_boundary(
        pipe, thr, pca_tr, pg_tr, y_tr, f"SVM boundary — train · {title}",
        FIG_DIR / "ca_vs_gnn_svm_boundary_train.png",
    )
    _plot_boundary(
        pipe, thr, pca_t, pg_t, y_t, f"SVM boundary — test · {title}",
        FIG_DIR / "ca_vs_gnn_svm_boundary_test.png",
    )

    out = {
        "method": "svm_2d(ca_score, gnn_score + poly feats); thr on decision_function (val)",
        "ca_seed_weights": w.tolist(),
        "baseline_logreg_router_test_f1": baseline,
        "test_ca_alone": ca_alone,
        "test_gnn_alone": gnn_alone,
        "best": {k: best[k] for k in best if k != "pipe"},
        "test_f1": float(report["macro_f1"]),
        "promote": bool(report["macro_f1"] > baseline),
        "top5_by_val": top5,
        "best_per_kernel_test": kernel_test,
        "report": report,
        "figures": {
            "train": str(FIG_DIR / "ca_vs_gnn_svm_boundary_train.png"),
            "test": str(FIG_DIR / "ca_vs_gnn_svm_boundary_test.png"),
        },
        "note": (
            "Seleção só na val. Sem certainty_ah. Comparar promote vs meta_router_ca_gnn.json."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary = {k: out[k] for k in out if k != "report"}
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(
        f"TEST macro_f1={report['macro_f1']:.4f} "
        f"(baseline LogReg router={baseline:.4f}, promote={out['promote']}) → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
