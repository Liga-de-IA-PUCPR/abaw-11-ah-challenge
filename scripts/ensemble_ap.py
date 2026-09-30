#!/usr/bin/env python3
"""Ensemble por média ponderada de AP + re-score honesto do meta-router.

Dois modos (FASE 0 do plano `docs/improvement_plan.md`):

``ap-weighted`` (default)
    Carrega as ``predictions.csv`` de vários membros (mesmo formato usado
    por ``scripts/meta_router_ca_gnn_face.py`` linhas ~85-106:
    ``outputs/<run>/eval_{train,val,test}/predictions.csv`` com colunas
    ``video_id,y_true,y_proba``), combina por **média ponderada por AP**
    (``weight_m = AP_m / Σ AP``, Eq. 3 do IISERB) com **limiar fixo 0.5**,
    e reporta Macro-F1/AP + IC por bootstrap (`src/eval/protocol.py`).
    Os pesos por AP são calculados no split de **referência** (``--ap-split``,
    default ``val``) e reaplicados sem reajuste no split de **avaliação**
    (``--eval-split``, default ``test``) — nenhum hiperparâmetro é escolhido
    olhando o split avaliado.

``rescore-router``
    Reavalia o meta-router CA⊕GNN de produção (ver
    ``references/meta_router_ca_gnn.md`` e ``scripts/meta_router_ca_gnn.py``)
    com limiar **fixo 0.5** em vez do limiar/roteamento calibrado na val
    (0.7454 no test com seleção conjunta val-tuned), para estimar quanto
    do ganho reportado é sobreajuste ao val de 124 vídeos.

    **Limitação documentada:** o router não é um modelo único treinável
    fold-a-fold — é uma seleção conjunta (pesos dos 7 seeds CA + LogReg de
    discordância + limiares + τ) feita inteiramente a partir do split
    train/val fixo. Não há OOF genuíno disponível para ele sem reimplementar
    o protocolo de seleção dentro de cada fold (fora do escopo da FASE 0).
    Este modo portanto **não roda o protocolo OOF** de
    ``src/eval/protocol.py``: ele reproduz o treino/seleção do router
    exatamente como ``scripts/meta_router_ca_gnn.py`` faz (train→pesos,
    train+val→LogReg, val→escolha de thr_ca/C/τ) e então recomputa o score
    combinado do **test já existente**, comparando:
      (a) Macro-F1 do router com o limiar/roteamento tal como calibrado
          (reportado como 0.7454); vs.
      (b) Macro-F1 do MESMO score combinado do router, mas com limiar
          fixo 0.5 (sem o roteamento por τ calibrado na val — usa-se
          `score_CA` re-roteada apenas pela discordância, limiar 0.5).
    O IC do bootstrap pareado (a - b) mostra quanto do ganho é atribuível
    à calibração no val pequeno.

Uso::

    uv run python scripts/ensemble_ap.py ap-weighted \\
        --member outputs/cross_attention/20260713_175242 \\
        --member outputs/hetero_gnn_contrastive/20260713_162753

    uv run python scripts/ensemble_ap.py rescore-router
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.protocol import (  # noqa: E402
    average_precision_score_,
    macro_f1_at_threshold,
    paired_bootstrap_macro_f1,
)
from src.logger import get_logger  # noqa: E402

log = get_logger("scripts.ensemble_ap")


# ==============================================================================
# Carregamento de predictions.csv (mesmo padrão de meta_router_ca_gnn_face.py)
# ==============================================================================


def _load_predictions(run_dir: Path, split: str) -> tuple[dict[str, float], dict[str, int]]:
    """Lê ``outputs/<run>/eval_{split}/predictions.csv`` → (scores, labels)."""
    path = run_dir / f"eval_{split}" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    scores: dict[str, float] = {}
    labels: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["video_id"]] = float(row["y_proba"])
            labels[row["video_id"]] = int(row["y_true"])
    return scores, labels


def _align_members(member_dirs: list[Path], split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Alinha vários membros num split: ``(ids, scores[n_members, n_ids], y[n_ids])``."""
    per_member = [_load_predictions(d, split) for d in member_dirs]
    ids = sorted(set.intersection(*[set(scores) for scores, _ in per_member]))
    if not ids:
        raise ValueError(f"nenhum video_id comum entre os membros no split {split!r}")
    labels_ref = per_member[0][1]
    y = np.array([labels_ref[vid] for vid in ids], dtype=np.int64)
    scores = np.array(
        [[per_member[m][0][vid] for vid in ids] for m in range(len(member_dirs))],
        dtype=np.float64,
    )
    return ids, scores, y


# ==============================================================================
# Modo 1 — ensemble ponderado por AP
# ==============================================================================


def ap_weighted_ensemble(
    member_dirs: list[Path],
    ap_split: str = "val",
    eval_split: str = "test",
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Combina membros por média ponderada de AP; avalia com limiar fixo.

    Os pesos ``weight_m = AP_m / Σ AP`` são calculados no split
    ``ap_split`` (nunca no split avaliado), e reaplicados sem reajuste no
    split ``eval_split``. Compara contra o melhor membro individual via
    bootstrap pareado.
    """
    _, scores_ap, y_ap = _align_members(member_dirs, ap_split)
    aps = np.array([average_precision_score_(y_ap, scores_ap[m]) for m in range(len(member_dirs))])
    if aps.sum() <= 0:
        raise ValueError("soma das APs é <= 0; não é possível ponderar")
    weights = aps / aps.sum()

    ids_eval, scores_eval, y_eval = _align_members(member_dirs, eval_split)
    ensemble_score = (weights[:, None] * scores_eval).sum(axis=0)

    ensemble_f1 = macro_f1_at_threshold(y_eval, ensemble_score, threshold=threshold)
    ensemble_ap = average_precision_score_(y_eval, ensemble_score)

    best_member_idx = int(np.argmax(aps))
    best_member_score = scores_eval[best_member_idx]
    best_member_f1 = macro_f1_at_threshold(y_eval, best_member_score, threshold=threshold)

    boot = paired_bootstrap_macro_f1(
        y_eval,
        ensemble_score,
        best_member_score,
        threshold=threshold,
        n_boot=n_boot,
        seed=seed,
    )

    result = {
        "members": [str(d) for d in member_dirs],
        "ap_split": ap_split,
        "eval_split": eval_split,
        "threshold": threshold,
        "n_videos": len(ids_eval),
        "weights": weights.tolist(),
        "aps_on_ap_split": aps.tolist(),
        "ensemble_macro_f1": ensemble_f1,
        "ensemble_ap": ensemble_ap,
        "best_member_idx": best_member_idx,
        "best_member_macro_f1": best_member_f1,
        "bootstrap_ensemble_vs_best_member": boot,
    }
    return result


# ==============================================================================
# Modo 2 — re-score honesto do meta-router CA⊕GNN
# ==============================================================================


def _select_best_router_candidate(
    router,
    weight_cands: list[np.ndarray],
    train_split: tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, str]],
    val_split: tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, str]],
    thr_gnn: float,
) -> dict | None:
    """Reproduz a busca de `scripts/meta_router_ca_gnn.py::main` (pesos, C, tau).

    ``train_split``/``val_split`` são as tuplas ``(ids, y, ca, gnn, question_type)``
    devolvidas por ``router._load_split`` — mesmo formato usado em produção.
    """
    from sklearn.linear_model import LogisticRegression

    ids_tr, y_tr, ca_tr, pg_tr, qt_tr = train_split
    ids_v, y_v, ca_v, pg_v, qt_v = val_split

    best: dict | None = None
    for w in weight_cands:
        pca_tr = (ca_tr * w).sum(1)
        pca_v = (ca_v * w).sum(1)
        thr_ca, _ = router._best_thr(pca_v, y_v)
        X_tr = router._features(pca_tr, pg_tr, ids_tr, qt_tr, thr_ca, thr_gnn)
        X_v = router._features(pca_v, pg_v, ids_v, qt_v, thr_ca, thr_gnn)
        mask_tr, pick_tr = router._pick_labels(pca_tr, pg_tr, y_tr, thr_ca, thr_gnn)
        if int(mask_tr.sum()) < 10 or len(np.unique(pick_tr[mask_tr])) < 2:
            continue
        for C in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
            clf.fit(X_tr[mask_tr], pick_tr[mask_tr])
            for tau in np.linspace(0.30, 0.70, 17):
                pred_v = router._route(clf, X_v, pca_v, pg_v, thr_ca, thr_gnn, float(tau))
                val_f1 = router._macro_f1(y_v, pred_v)
                n_swap = int((pred_v != (pca_v >= thr_ca)).sum())
                cand = {
                    "val_f1": val_f1,
                    "n_swap_val": n_swap,
                    "C": float(C),
                    "tau": float(tau),
                    "thr_ca": float(thr_ca),
                    "w": w,
                    "clf": clf,
                }
                if (
                    best is None
                    or val_f1 > best["val_f1"]
                    or (abs(val_f1 - best["val_f1"]) < 1e-12 and n_swap > best["n_swap_val"])
                ):
                    best = cand
    return best


def rescore_router(
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Reproduz o treino/seleção do router e recomputa o test com limiar fixo.

    Ver docstring do módulo para a limitação: não é OOF genuíno (o router é
    uma seleção conjunta sobre um único split train/val, não um modelo
    fold-a-fold). Requer ``scripts/meta_router_ca_gnn.py`` disponível no
    disco (ele reexporta as funções internas usadas aqui via
    ``importlib``); se o arquivo não existir (ex.: checkout parcial),
    levanta ``FileNotFoundError`` com uma mensagem clara em vez de falhar
    silenciosamente.
    """
    router_path = ROOT / "scripts" / "meta_router_ca_gnn.py"
    if not router_path.exists():
        raise FileNotFoundError(
            f"{router_path} não encontrado — modo rescore-router precisa do script "
            "de produção do meta-router (scripts/meta_router_ca_gnn.py) no disco. "
            "Rode este script a partir de um checkout completo do repositório."
        )
    spec = importlib.util.spec_from_file_location("_meta_router_ca_gnn", router_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"não foi possível carregar {router_path}")
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)

    train_split = router._load_split("train")
    val_split = router._load_split("val")
    ids_t, y_t, ca_t, pg_t, qt_t = router._load_split("test")
    _, y_tr, ca_tr, _, _ = train_split

    import json as _json

    state = _json.loads((ROOT / router.GNN_RUN / "trainer_state.json").read_text(encoding="utf-8"))
    thr_gnn = float(state["threshold"])

    weight_cands = router._candidate_weights(ca_tr, y_tr, seed=seed)
    best = _select_best_router_candidate(router, weight_cands, train_split, val_split, thr_gnn)
    if best is None:
        raise RuntimeError("nenhum candidato válido para o roteador")

    w = best["w"]
    thr_ca = best["thr_ca"]
    clf = best["clf"]
    tau = best["tau"]

    pca_t = (ca_t * w).sum(1)
    X_t = router._features(pca_t, pg_t, ids_t, qt_t, thr_ca, thr_gnn)

    # (a) roteamento calibrado na val (limiar/τ escolhidos na val — produção).
    pred_t_calibrated = router._route(clf, X_t, pca_t, pg_t, thr_ca, thr_gnn, tau)
    f1_calibrated = router._macro_f1(y_t, pred_t_calibrated)

    # Score combinado (mesmo usado no JSON de produção): score_CA, trocado
    # pelo GNN nas discordâncias onde P(GNN) >= tau.
    agree = (pca_t >= thr_ca) == (pg_t >= thr_gnn)
    combined_score = pca_t.copy()
    if (~agree).any():
        p_gnn = clf.predict_proba(X_t[~agree])[:, 1]
        use_g = p_gnn >= tau
        combined_score = combined_score.copy()
        combined_score[~agree] = np.where(use_g, pg_t[~agree], pca_t[~agree])

    # (b) MESMO score combinado, mas limiar fixo 0.5 (sem thr_ca calibrado).
    f1_fixed_threshold = macro_f1_at_threshold(y_t, combined_score, threshold=threshold)
    ap_combined = average_precision_score_(y_t, combined_score)

    boot = paired_bootstrap_macro_f1(
        y_t,
        (pred_t_calibrated).astype(np.float64),  # score binário 0/1 do roteamento calibrado
        combined_score,
        threshold=threshold,
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "n_videos": len(ids_t),
        "threshold_fixed": threshold,
        "router_calibrated_macro_f1": f1_calibrated,
        "router_fixed_threshold_macro_f1": f1_fixed_threshold,
        "router_combined_score_ap": ap_combined,
        "gap_calibrated_minus_fixed": f1_calibrated - f1_fixed_threshold,
        "bootstrap_calibrated_vs_fixed": boot,
        "note": (
            "Não é OOF genuíno: pesos/LogReg/thr_ca/tau vêm da seleção conjunta "
            "train→val do router de produção (scripts/meta_router_ca_gnn.py). "
            "Compara o MESMO score combinado do test com o limiar/roteamento "
            "calibrado na val (produção) vs. limiar fixo 0.5."
        ),
    }


# ==============================================================================
# CLI
# ==============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_ap = sub.add_parser("ap-weighted", help="ensemble por média ponderada de AP")
    p_ap.add_argument(
        "--member",
        action="append",
        required=True,
        dest="members",
        help="diretório outputs/<run> de um membro (repetível, >=2 membros)",
    )
    p_ap.add_argument("--ap-split", default="val", help="split usado p/ calcular os pesos AP")
    p_ap.add_argument("--eval-split", default="test", help="split avaliado (limiar fixo)")
    p_ap.add_argument("--threshold", type=float, default=0.5)
    p_ap.add_argument("--n-boot", type=int, default=1000)
    p_ap.add_argument("--seed", type=int, default=0)

    p_rr = sub.add_parser(
        "rescore-router", help="re-score do meta-router CA⊕GNN com limiar fixo 0.5"
    )
    p_rr.add_argument("--threshold", type=float, default=0.5)
    p_rr.add_argument("--n-boot", type=int, default=1000)
    p_rr.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.mode == "ap-weighted":
        if len(args.members) < 2:
            parser.error("ap-weighted precisa de pelo menos 2 --member")
        result = ap_weighted_ensemble(
            [Path(m) for m in args.members],
            ap_split=args.ap_split,
            eval_split=args.eval_split,
            threshold=args.threshold,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    else:
        result = rescore_router(threshold=args.threshold, n_boot=args.n_boot, seed=args.seed)

    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
