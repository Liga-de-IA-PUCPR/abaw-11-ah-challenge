"""``mode=route`` — MoERouter sobre membros avaliados no protocolo OOF (Rodada 6).

Entrada: ``route.members=[<run OOF>, ...]`` (``outputs/oof/<exp>/<ts>``, mesmas dobras — mesma
``oof.seed``/``oof.n_splits``). De cada membro: ``oof_predictions.csv`` (logit), e — se
houver — ``oof_embeddings.npy`` (vetor pré-logit) e ``pred_<split>_folds.npz``.

1. Avaliação honesta do roteador nas MESMAS dobras dos membros: dobra k → treina o
   :class:`~src.models.moe_router.MoERouter` nas demais (early stopping num holdout interno
   por participante) e prediz k. Como as predições dos membros já são OOF, isto é stacking
   sem vazamento.
2. Métricas com τ fixo (0.5) + gate pareado vs. o MELHOR membro (AP OOF) e vs. a MÉDIA simples
   dos membros; uso médio de cada membro pelo roteador (quem ele escolhe).
3. Roteador final (todas as linhas OOF) aplicado aos splits de ``route.predict_splits``: para
   cada dobra k, combina as saídas dos modelos da dobra k dos membros (mesmo espaço de
   embedding que o roteador viu no OOF) e tira a média das dobras.

Saída em ``outputs/route/<route.name>/<timestamp>/``: ``route_oof.csv``, ``route_metrics.json``,
``pred_<split>.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.eval.protocol import (
    bootstrap_ci_macro_f1,
    make_group_folds,
    paired_bootstrap_ap,
    paired_bootstrap_macro_f1,
)
from src.logger import get_logger
from src.models.moe_router import MoERouter, RouterInputs, to_logit
from src.pipeline.oof import _metrics, _timestamp

log = get_logger("pipeline.route")


def run_route(cfg: DictConfig) -> dict[str, Any]:
    """Avalia (OOF) e aplica o MoERouter sobre ``cfg.route.members``."""
    r = cfg.route
    members = [Path(str(m)) for m in r.members]
    if len(members) < 2:
        raise ValueError("route.members: forneça >= 2 runs OOF (outputs/oof/<exp>/<ts>).")
    table, inputs = load_members(members, bool(r.use_embeddings))
    names = [f"{m.parent.name}/{m.name}" for m in members]
    y, folds, tau = table["y_true"].to_numpy(), table["fold"].to_numpy(), float(r.threshold)
    out_dir = Path(cfg.data.paths.output_root) / "route" / str(r.name) / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    proba = np.zeros(len(table))
    weights = np.zeros((len(table), len(members)))
    for k in np.unique(folds):
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        router = _fit(r, table.iloc[train], inputs.take(train), int(cfg.seed) + int(k))
        proba[test], weights[test] = router.predict(inputs.take(test))

    member_proba = 1.0 / (1.0 + np.exp(-inputs.logits.astype(np.float64)))
    mean_proba = member_proba.mean(axis=1)
    member_metrics = {n: _metrics(y, member_proba[:, i], tau) for i, n in enumerate(names)}
    best = max(member_metrics, key=lambda n: member_metrics[n]["ap"])
    best_proba = member_proba[:, names.index(best)]
    n_boot = int(r.n_boot)
    report: dict[str, Any] = {
        "members": names,
        "use_embeddings": bool(r.use_embeddings),
        "threshold": tau,
        "router": {
            **_metrics(y, proba, tau),
            "macro_f1_ci": bootstrap_ci_macro_f1(y, proba, tau, n_boot),
        },
        "mean_of_members": _metrics(y, mean_proba, tau),
        "members_metrics": member_metrics,
        "best_member": best,
        "gate_vs_best": _gate(y, proba, best_proba, tau, n_boot),
        "gate_vs_mean": _gate(y, proba, mean_proba, tau, n_boot),
        "usage": {n: float(weights[:, i].mean()) for i, n in enumerate(names)},
        "argmax_share": {n: float(np.mean(weights.argmax(1) == i)) for i, n in enumerate(names)},
    }
    oof = table.assign(y_proba=proba, **{f"w_{i}": weights[:, i] for i in range(len(names))})
    oof.to_csv(out_dir / "route_oof.csv", index=False)

    final = _fit(r, table, inputs, int(cfg.seed))
    for split in r.get("predict_splits") or []:
        pred = predict_split(final, members, split, bool(r.use_embeddings))
        if pred is not None:
            pred.to_csv(out_dir / f"pred_{split}.csv", index=False)
    (out_dir / "route_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _log_summary(report)
    return {"out_dir": str(out_dir), **report["router"]}


def load_members(members: list[Path], use_embeddings: bool) -> tuple[pd.DataFrame, RouterInputs]:
    """Alinha os ``oof_predictions.csv`` (mesmos vídeos E mesmas dobras) e os embeddings."""
    tables = [pd.read_csv(m / "oof_predictions.csv").set_index("video_id") for m in members]
    base = tables[0].sort_index()
    logits, embeddings = [], []
    for m, t in zip(members, tables, strict=True):
        t_aligned = t.reindex(base.index)
        if t_aligned["y_proba"].isna().any() or not (t_aligned["fold"] == base["fold"]).all():
            raise ValueError(
                f"Membro {m}: vídeos/dobras diferentes do 1º membro — rode todos com o mesmo "
                "oof.seed/oof.n_splits/oof.splits."
            )
        logits.append(to_logit(t_aligned["y_proba"].to_numpy()))
        emb_file = m / "oof_embeddings.npy"
        if use_embeddings and emb_file.exists():
            emb = np.load(emb_file)
            embeddings.append(emb[t.index.get_indexer(base.index)])
        else:
            embeddings.append(None)
    table = base.reset_index()[["video_id", "participant_id", "y_true", "fold"]]
    return table, RouterInputs(np.stack(logits, axis=1), embeddings)


def predict_split(
    router: MoERouter, members: list[Path], split: str, use_embeddings: bool
) -> pd.DataFrame | None:
    """Média das dobras do roteador aplicado às saídas por dobra dos membros num split."""
    files = [m / f"pred_{split}_folds.npz" for m in members]
    if not all(f.exists() for f in files):
        log.warning(f"Split '{split}': algum membro sem pred_{split}_folds.npz — pulado.")
        return None
    packs = [np.load(f, allow_pickle=False) for f in files]
    ids = packs[0]["video_ids"]
    order = [pd.Index(p["video_ids"]).get_indexer(ids) for p in packs]
    n_folds = sum(1 for key in packs[0].files if key.startswith("proba_fold"))
    probas = []
    for k in range(n_folds):
        logits = np.stack(
            [to_logit(p[f"proba_fold{k}"][o]) for p, o in zip(packs, order, strict=True)], 1
        )
        embs = [
            p[f"embedding_fold{k}"][o]
            if use_embeddings and f"embedding_fold{k}" in p.files
            else None
            for p, o in zip(packs, order, strict=True)
        ]
        embs = [e if router.net.has_emb[i] else None for i, e in enumerate(embs)]
        probas.append(router.predict(RouterInputs(logits, embs))[0])
    return pd.DataFrame({"video_id": ids, "y_proba": np.mean(probas, axis=0)})


def _fit(r: DictConfig, table: pd.DataFrame, inputs: RouterInputs, seed: int) -> MoERouter:
    """Treina o roteador nas linhas de ``table`` com holdout interno por participante."""
    df = table.rename(columns={"y_true": "label"}).reset_index(drop=True)
    inner = make_group_folds(df, n_splits=max(2, round(1 / float(r.inner_val_frac))), seed=seed)
    fit, val = np.flatnonzero(inner != 0), np.flatnonzero(inner == 0)
    y = df["label"].to_numpy().astype(np.float32)
    router = MoERouter(
        proj_dim=int(r.proj_dim),
        dropout=float(r.dropout),
        lr=float(r.lr),
        weight_decay=float(r.weight_decay),
        max_epochs=int(r.max_epochs),
        patience=int(r.patience),
        seed=seed,
    )
    return router.fit(inputs.take(fit), y[fit], inputs.take(val), y[val])


def _gate(y, a, b, tau: float, n_boot: int) -> dict[str, Any]:
    f1 = paired_bootstrap_macro_f1(y, a, b, threshold=tau, n_boot=n_boot)
    ap = paired_bootstrap_ap(y, a, b, n_boot=n_boot)
    verdict = "melhora" if f1["ci_low"] > 0 else "piora" if f1["ci_high"] < 0 else "empate"
    return {"macro_f1": f1, "ap": ap, "verdict": verdict}


def _log_summary(report: dict[str, Any]) -> None:
    r = report["router"]
    log.info(
        f"MoERouter OOF: Macro-F1@{report['threshold']}={r['macro_f1']:.4f} AP={r['ap']:.4f} · "
        f"média dos membros F1={report['mean_of_members']['macro_f1']:.4f} "
        f"AP={report['mean_of_members']['ap']:.4f} · melhor membro {report['best_member']}"
    )
    for key in ("gate_vs_best", "gate_vs_mean"):
        g = report[key]
        log.info(
            f"{key}: ΔF1={g['macro_f1']['observed_diff']:+.4f} "
            f"[{g['macro_f1']['ci_low']:+.4f}, {g['macro_f1']['ci_high']:+.4f}] · "
            f"ΔAP={g['ap']['observed_diff']:+.4f} → {g['verdict']}"
        )
    log.info(f"Uso médio dos membros: {report['usage']}")
