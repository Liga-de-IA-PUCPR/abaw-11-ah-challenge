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
3. Roteador final (todas as linhas OOF) aplicado aos nomes de ``route.predict_splits`` (splits
   de ``oof.predict_splits`` ou Parquets de ``oof.predict_parquets``, ex. o private test): para
   cada dobra k, combina as saídas dos modelos da dobra k dos membros (mesmo espaço de
   embedding que o roteador viu no OOF) e tira a média das dobras.
4. Rodada 6, opt-in: ``route.measure_splits=[test]`` é a medição ÚNICA no public test (roteador
   × cada membro × média); ``route.submit_split=external`` escreve a submissão oficial
   (``out``, ``submission_reference``, ``submission_probabilities``, τ fixo).

Saída em ``outputs/route/<route.name>/<timestamp>/`` (a pasta do run Hydra, com ``.hydra/`` e
``main.log``): ``route_oof.csv``, ``route_metrics.json``, ``plots/``, ``pred_<nome>.csv`` e, se
pedida, a submissão.
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
from src.outputs.checkpoint import hydra_run_dir
from src.pipeline.oof import _metrics, _timestamp, save_plots

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
    out_dir = hydra_run_dir() or (
        Path(cfg.data.paths.output_root) / "route" / str(r.name) / _timestamp()
    )
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
    preds: dict[str, pd.DataFrame] = {}
    for split in r.get("predict_splits") or []:
        pred = predict_split(final, members, split, bool(r.use_embeddings))
        if pred is not None:
            pred.to_csv(out_dir / f"pred_{split}.csv", index=False)
            preds[split] = pred
    for split in r.get("measure_splits") or []:  # medição ÚNICA no public test (Rodada 6)
        report[f"measure_{split}"] = measure_split(cfg, members, names, preds, split, tau, n_boot)
    if r.get("submit_split"):
        report["submission"] = str(write_route_submission(cfg, preds, str(r.submit_split), out_dir))
    (out_dir / "route_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    save_plots(out_dir, y, proba, tau)
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


def measure_split(
    cfg: DictConfig,
    members: list[Path],
    names: list[str],
    preds: dict[str, pd.DataFrame],
    split: str,
    tau: float,
    n_boot: int,
) -> dict[str, Any]:
    """Métricas num split ROTULADO do Parquet de treino: roteador × cada membro × média.

    É a medição única da Rodada 6 — opt-in (``route.measure_splits``), nunca automática.
    """
    from src.data.datasets import video_table

    if split not in preds:
        raise ValueError(f"route.measure_splits: '{split}' precisa estar em route.predict_splits")
    labels = video_table(cfg.data.paths.parquet_path, [split]).set_index("video_id")["label"]
    router = preds[split].set_index("video_id")["y_proba"]
    ids = [v for v in router.index if v in labels.index]
    if not ids or (labels.loc[ids] < 0).any():
        raise ValueError(f"route.measure_splits: '{split}' não tem rótulos no Parquet de treino")
    y = labels.loc[ids].to_numpy()
    member_p = {
        n: pd.read_csv(m / f"pred_{split}.csv").set_index("video_id")["y_proba"].loc[ids].to_numpy()
        for n, m in zip(names, members, strict=True)
    }
    out = {
        "n_videos": len(ids),
        "router": {
            **_metrics(y, router.loc[ids].to_numpy(), tau),
            "macro_f1_ci": bootstrap_ci_macro_f1(y, router.loc[ids].to_numpy(), tau, n_boot),
        },
        "mean_of_members": _metrics(y, np.mean(list(member_p.values()), axis=0), tau),
        "members": {n: _metrics(y, p, tau) for n, p in member_p.items()},
    }
    log.info(
        f"Medição única em '{split}' ({len(ids)} vídeos): MoERouter "
        f"Macro-F1@{tau}={out['router']['macro_f1']:.4f} AP={out['router']['ap']:.4f} · "
        f"média dos membros F1={out['mean_of_members']['macro_f1']:.4f}"
    )
    return out


def write_route_submission(
    cfg: DictConfig, preds: dict[str, pd.DataFrame], split: str, out_dir: Path
) -> Path:
    """Submissão oficial a partir das probas do roteador em ``split`` (τ = ``route.threshold``)."""
    from src.outputs.submission import read_reference_order, write_submission

    if split not in preds:
        raise ValueError(f"route.submit_split: '{split}' precisa estar em route.predict_splits")
    scores = dict(zip(preds[split]["video_id"].astype(str), preds[split]["y_proba"], strict=True))
    tau = float(cfg.route.threshold)
    return write_submission(
        {v: int(p >= tau) for v, p in scores.items()},
        Path(cfg.get("out") or out_dir / f"submission_{split}.txt"),
        order=read_reference_order(cfg.get("submission_reference")),
        probabilities=scores if cfg.get("submission_probabilities") else None,
    )


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
