"""``mode=oof`` — protocolo OOF do plano MoE: a régua de TODA rodada (gate de decisão).

Para o modelo de ``cfg.model`` (qualquer modelo do registry — RF, cross-attention, GNN,
``moe_fusion``), sobre os vídeos de ``oof.splits`` (default train+val = 902):

    1. 5 dobras ``StratifiedGroupKFold`` por participante (:func:`make_group_folds`), com a
       MESMA semente em todas as rodadas → comparações pareadas vídeo a vídeo.
    2. Dobra k: treina nas outras dobras com um holdout INTERNO por participante
       (``oof.inner_val_frac``) p/ early stopping/checkpoint; prediz a dobra k (OOF), os
       splits de ``oof.predict_splits`` (ex.: test → média das 5 dobras = bagging, gravada
       SEM métricas: o public test é medido uma única vez, na consolidação) e os Parquets
       inteiros de ``oof.predict_parquets`` (ex.: ``{external: <private test>}``).
    3. Macro-F1 com limiar FIXO (``oof.threshold``, 0.5), AP e IC por bootstrap; com
       ``oof.baseline=<run OOF>``, o gate pareado (Δ Macro-F1 e Δ AP, bootstrap pareado).

Valores de config com ``{fold}`` são formatados por dobra — ex.: o MoE da Rodada 2 parte do
texto fine-tunado da MESMA dobra da Rodada 1:
``"model.branches.text.init_from='outputs/oof/moe-r1-text/<ts>/fold{fold}'"`` (entre aspas
na CLI: o Hydra lê ``{`` como início de dict).

Saída em ``outputs/oof/<experiment_name>/<timestamp>/`` (a pasta do run Hydra, com ``.hydra/``
e ``main.log``): ``oof_predictions.csv``, ``oof_embeddings.npy`` (linhas = csv),
``oof_metrics.json``, ``plots/`` (:func:`save_plots`), ``pred_<nome>.csv`` (+
``pred_<nome>_folds.npz``, por split/Parquet extra) e ``fold<k>/`` (run dir de cada dobra).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.eval.protocol import (
    average_precision_score_,
    bootstrap_ci_macro_f1,
    macro_f1_at_threshold,
    make_group_folds,
    paired_bootstrap_ap,
    paired_bootstrap_macro_f1,
)
from src.logger import get_logger
from src.outputs.checkpoint import hydra_run_dir

log = get_logger("pipeline.oof")


def run_oof(cfg: DictConfig) -> dict[str, Any]:
    """Roda o protocolo OOF do modelo de ``cfg`` e grava predições + métricas + gate."""
    from src.data.datasets import load_split, load_videos, parquet_video_ids, video_table
    from src.models.registry import get_family

    o = cfg.oof
    family = get_family(str(cfg.model.name))
    videos = video_table(cfg.data.paths.parquet_path, list(o.splits))
    folds = make_group_folds(videos, n_splits=int(o.n_splits), seed=int(o.seed))
    out_dir = hydra_run_dir() or (
        Path(cfg.data.paths.output_root) / "oof" / str(cfg.experiment_name) / _timestamp()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"OOF de '{cfg.model.name}' ({family}): {len(videos)} vídeos → {out_dir}")

    pool = load_videos(cfg, set(videos["video_id"]), family=family)  # Parquet lido 1× só
    extra = {s: load_split(cfg, s, family=family) for s in (o.get("predict_splits") or [])}
    for name, pq in (o.get("predict_parquets") or {}).items():  # ex.: private test externo
        if name in extra:
            raise ValueError(f"oof.predict_parquets: {name!r} repete um nome de predict_splits")
        extra[name] = load_videos(cfg, parquet_video_ids(pq), family=family, parquet_path=pq)
    proba = np.full(len(videos), np.nan)
    emb_parts: dict[int, np.ndarray] = {}
    extra_out: dict[str, list[dict[str, np.ndarray]]] = {s: [] for s in extra}
    per_fold: list[dict[str, float]] = []
    for k in range(int(o.n_splits)):
        test_idx = np.flatnonzero(folds == k)
        out, extra_k = _run_fold(cfg, k, videos, folds, pool, extra, family, out_dir)
        proba[test_idx] = out["proba"]
        if "embedding" in out:
            emb_parts[k] = out["embedding"]
        per_fold.append(_metrics(videos["label"].to_numpy()[test_idx], out["proba"], o.threshold))
        log.info(f"Dobra {k}: Macro-F1={per_fold[-1]['macro_f1']:.4f} AP={per_fold[-1]['ap']:.4f}")
        for split, split_out in extra_k.items():
            extra_out[split].append(split_out)

    videos = videos.assign(fold=folds, y_proba=proba)
    videos.rename(columns={"label": "y_true"}).to_csv(out_dir / "oof_predictions.csv", index=False)
    if len(emb_parts) == int(o.n_splits):  # vetores pré-logit → MoERouter
        emb = np.zeros((len(videos), emb_parts[0].shape[1]), dtype=np.float32)
        for k, part in emb_parts.items():
            emb[folds == k] = part
        np.save(out_dir / "oof_embeddings.npy", emb)
    for split, outs in extra_out.items():
        _save_bagged(out_dir, split, outs)

    report = _report(cfg, videos, per_fold)
    (out_dir / "oof_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    save_plots(out_dir, videos["label"].to_numpy(), proba, float(o.threshold))
    _log_summary(report)
    return {"out_dir": str(out_dir), **{k: report[k] for k in ("macro_f1", "ap")}}


def _run_fold(cfg, k, videos, folds, pool, extra, family, out_dir):
    """Treina a dobra ``k`` (holdout interno p/ early stopping) e prediz a dobra + extras.

    Devolve ``(saídas na ordem de videos[folds == k], {split: saídas})``.
    """
    from src.data.datasets import as_loader
    from src.models.registry import create_model
    from src.training.factory import create_trainer

    o = cfg.oof
    fold_cfg = _format_fold(cfg, k)
    train_df = videos.iloc[np.flatnonzero(folds != k)].reset_index(drop=True)
    n_inner = max(2, round(1 / float(o.inner_val_frac)))
    inner = make_group_folds(train_df, n_splits=n_inner, seed=int(o.seed) + k + 1)
    fit_ids, val_ids = (
        set(train_df["video_id"][inner != 0]),
        set(train_df["video_id"][inner == 0]),
    )
    test_ids = videos["video_id"].iloc[np.flatnonzero(folds == k)]
    log.info(f"=== Dobra {k}: treino {len(fit_ids)} · val {len(val_ids)} · OOF {len(test_ids)} ===")

    model, _ = create_model(fold_cfg.model.name, fold_cfg.model)
    trainer = create_trainer(family, model=model, cfg=fold_cfg)
    trainer.output_dir = str(out_dir / f"fold{k}")
    trainer.fit(
        as_loader(fold_cfg, pool.subset(fit_ids), family, "train"),
        as_loader(fold_cfg, pool.subset(val_ids), family, "val"),
    )
    trainer.save(out_dir / f"fold{k}")

    def predict(view) -> dict[str, np.ndarray]:
        return _outputs(trainer, family, as_loader(fold_cfg, view, family, "eval"))

    out = predict(pool.subset(set(test_ids)))
    order = pd.Index(out["video_ids"]).get_indexer(test_ids)
    extra_out = {s: predict(v) for s, v in extra.items()}
    _finish_wandb()
    return {key: value[order] for key, value in out.items()}, extra_out


def _report(cfg: DictConfig, videos: pd.DataFrame, per_fold: list[dict]) -> dict[str, Any]:
    """Métricas OOF com τ fixo + IC por bootstrap (+ gate vs ``oof.baseline``)."""
    o, tau = cfg.oof, float(cfg.oof.threshold)
    y, proba = videos["label"].to_numpy(), videos["y_proba"].to_numpy()
    report: dict[str, Any] = {
        "experiment": str(cfg.experiment_name),
        "model": str(cfg.model.name),
        "n_videos": len(videos),
        "splits": list(o.splits),
        "n_splits": int(o.n_splits),
        "seed": int(o.seed),
        "threshold": tau,
        **_metrics(y, proba, tau),
        "macro_f1_ci": bootstrap_ci_macro_f1(y, proba, tau, int(o.n_boot)),
        "per_fold": per_fold,
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
    }
    if o.get("baseline"):
        report["gate"] = gate(videos, Path(str(o.baseline)), tau, int(o.n_boot))
    return report


# ==============================================================================
# Gate pareado (rodada nova vs. referência, mesmos vídeos)
# ==============================================================================


def gate(videos: pd.DataFrame, baseline_dir: Path, threshold: float, n_boot: int) -> dict[str, Any]:
    """Δ Macro-F1@τ e Δ AP (novo − referência) com IC por bootstrap pareado + veredito.

    ``melhora`` se o IC 95% do Δ Macro-F1 fica todo acima de 0, ``piora`` se todo abaixo,
    ``empate`` caso contrário (o AP vai junto como régua livre de limiar).
    """
    base = pd.read_csv(baseline_dir / "oof_predictions.csv")
    merged = videos.merge(base[["video_id", "y_proba"]], on="video_id", suffixes=("", "_base"))
    if len(merged) != len(videos):
        log.warning(f"Gate: {len(videos) - len(merged)} vídeos sem par na referência (ignorados).")
    y, new, ref = merged["label"], merged["y_proba"], merged["y_proba_base"]
    f1 = paired_bootstrap_macro_f1(y, new, ref, threshold=threshold, n_boot=n_boot)
    ap = paired_bootstrap_ap(y, new, ref, n_boot=n_boot)
    verdict = "melhora" if f1["ci_low"] > 0 else "piora" if f1["ci_high"] < 0 else "empate"
    return {
        "baseline": str(baseline_dir),
        "n_paired": len(merged),
        "macro_f1": f1,
        "ap": ap,
        "verdict": verdict,
    }


def compare_runs(
    run_a: str | Path, run_b: str | Path, threshold: float = 0.5, n_boot: int = 1000
) -> dict[str, Any]:
    """Gate pareado entre dois runs OOF quaisquer (A − B) nos vídeos em comum — ex.: o MoE
    contra o cross-attention do artigo. Exige as mesmas dobras (mesma ``oof.seed``)."""
    videos = pd.read_csv(Path(run_a) / "oof_predictions.csv").rename(columns={"y_true": "label"})
    return {"run": str(run_a), **gate(videos, Path(run_b), threshold, n_boot)}


# ==============================================================================
# Helpers
# ==============================================================================


def _outputs(trainer, family: str, data) -> dict[str, np.ndarray]:
    """``video_ids``/``proba`` (+ ``embedding`` etc. no caminho lightning) de um split."""
    if family == "lightning":
        return trainer.predict_outputs(data)
    scores = trainer.predict_scores(data)
    ids = np.asarray(list(scores))
    return {"video_ids": ids, "proba": np.asarray([scores[v] for v in ids], dtype=np.float64)}


def _metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    threshold = float(threshold)
    return {
        "macro_f1": macro_f1_at_threshold(y, proba, threshold),
        "ap": average_precision_score_(y, proba),
        "pos_rate_pred": float(np.mean(proba >= threshold)),
        "pos_rate_true": float(np.mean(y)),
    }


def _save_bagged(out_dir: Path, split: str, outs: list[dict[str, np.ndarray]]) -> None:
    """Média das dobras num split extra (sem métricas) + saídas por dobra (MoERouter)."""
    ids = np.asarray(sorted(outs[0]["video_ids"]))
    aligned = [
        {k: v[pd.Index(o["video_ids"]).get_indexer(ids)] for k, v in o.items()} for o in outs
    ]
    proba = np.mean([a["proba"] for a in aligned], axis=0)
    pd.DataFrame({"video_id": ids, "y_proba": proba}).to_csv(
        out_dir / f"pred_{split}.csv", index=False
    )
    arrays = {f"proba_fold{k}": a["proba"] for k, a in enumerate(aligned)}
    arrays |= {
        f"embedding_fold{k}": a["embedding"] for k, a in enumerate(aligned) if "embedding" in a
    }
    np.savez(out_dir / f"pred_{split}_folds.npz", video_ids=ids, **arrays)


def _format_fold(cfg: DictConfig, fold: int) -> DictConfig:
    """Cópia da config com ``{fold}`` formatado em todo valor string (ex.: ``init_from``)."""

    def fmt(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: fmt(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fmt(v) for v in node]
        return node.replace("{fold}", str(fold)) if isinstance(node, str) else node

    return OmegaConf.create(fmt(OmegaConf.to_container(cfg, resolve=True)))


def save_plots(out_dir: Path, y: np.ndarray, proba: np.ndarray, threshold: float) -> None:
    """``plots/`` do run, os mesmos do evaluate: matriz de confusão com τ, ROC, PR e a curva
    limiar × Macro-F1 com o τ fixo marcado. Roda depois das predições/métricas gravadas e
    nunca derruba o run (horas de GPU) — falha vira aviso."""
    from src.outputs.reporter import Reporter
    from src.training.aggregation import threshold_curve

    try:
        rep = Reporter(out_dir)
        rep.plot_confusion_matrix(y, (proba >= threshold).astype(int))
        rep.plot_roc_curve(y, proba)
        rep.plot_precision_recall(y, proba)
        grid, f1s = threshold_curve(y, proba)
        rep.plot_threshold_curve(
            grid,
            f1s,
            threshold,
            label="τ fixo",
            ylabel="Macro-F1 (OOF, nível de vídeo)",
            title="Limiar × Macro-F1 (OOF)",
        )
    except Exception as exc:  # noqa: BLE001 — plots nunca derrubam o run
        log.warning(f"Plots pulados em {out_dir}: {exc}")


def plot_run(run_dir: str | Path) -> Path:
    """(Re)gera ``plots/`` de um run OOF ou route já gravado (ex.: runs de antes dos plots)."""
    run_dir = Path(run_dir)
    for preds, metrics in (
        ("oof_predictions.csv", "oof_metrics.json"),
        ("route_oof.csv", "route_metrics.json"),
    ):
        if (run_dir / preds).exists():
            df = pd.read_csv(run_dir / preds)
            m = run_dir / metrics
            tau = json.loads(m.read_text()).get("threshold", 0.5) if m.exists() else 0.5
            save_plots(run_dir, df["y_true"].to_numpy(), df["y_proba"].to_numpy(), float(tau))
            return run_dir / "plots"
    raise FileNotFoundError(f"{run_dir}: sem oof_predictions.csv nem route_oof.csv")


def _finish_wandb() -> None:
    """Fecha o run W&B da dobra: senão a dobra seguinte REUSA o run aberto (o WandbLogger
    adota o ``wandb.run`` ativo) e as 5 dobras gravam no mesmo run, em ``fold0/wandb``."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()


def _timestamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def _log_summary(report: dict[str, Any]) -> None:
    ci = report["macro_f1_ci"]
    log.info(
        f"OOF {report['experiment']}: Macro-F1@{report['threshold']}={report['macro_f1']:.4f} "
        f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] · AP={report['ap']:.4f} · "
        f"positivos preditos {report['pos_rate_pred']:.1%} (reais {report['pos_rate_true']:.1%})"
    )
    g = report.get("gate")
    if g:
        f1, ap = g["macro_f1"], g["ap"]
        log.info(
            f"Gate vs {g['baseline']}: ΔF1={f1['observed_diff']:+.4f} "
            f"[{f1['ci_low']:+.4f}, {f1['ci_high']:+.4f}] p={f1['p_value']:.3f} · "
            f"ΔAP={ap['observed_diff']:+.4f} [{ap['ci_low']:+.4f}, {ap['ci_high']:+.4f}] "
            f"→ {g['verdict']}"
        )
