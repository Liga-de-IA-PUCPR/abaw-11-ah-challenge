"""Relatórios e plots locais do pipeline BAH a nível de vídeo (SEMPRE roda).

É o **fallback offline** do tracking: independe do W&B e dos pacotes neurais.
Usa apenas sklearn + matplotlib (deps core) — funciona para AS DUAS famílias
(``random_forest`` e ``cross_attention``), pois consome arrays numpy de predição
a nível de vídeo, não objetos de modelo.

Gera:
- ``metrics.json`` : dump estruturado (métricas + config + limiar + agregação).
- ``results.txt``  : relatório legível (Macro-F1, AP, classification_report).
- Plots em ``plots/``:
    1. Matriz de confusão **a nível de vídeo**.
    2. Top-k importâncias de features do RandomForest (pulado p/ cross_attention).
    3. Curva Precision–Recall + Average Precision (classe positiva A/H).
    4. Curva limiar × Macro-F1 (saída da calibração — RF ou sigmoid do neural).

Todos os plots degradam graciosamente: insumo ausente → plot pulado com aviso,
nunca erro.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger

log = get_logger("outputs.reporter")

CLASS_NAMES: list[str] = ["sem_A/H", "com_A/H"]


class Reporter:
    """Gera relatórios JSON/TXT e plots para um run do pipeline BAH.

    Usage:
        rep = Reporter(output_dir="outputs/random_forest/20260624_101500")
        rep.save_metrics_json(metrics, config, threshold, aggregation_method)
        rep.save_results_txt(metrics)
        rep.plot_confusion_matrix(y_true, y_pred)
        rep.plot_feature_importances(importances, feature_names, top_k=30)  # RF
        rep.plot_precision_recall(y_true, y_proba)
        rep.plot_threshold_curve(thresholds, scores, best_threshold)
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Relatórios textuais
    # ==========================================================================

    def save_metrics_json(
        self,
        metrics: dict[str, Any],
        config: dict[str, Any],
        threshold: float,
        aggregation_method: str,
    ) -> Path:
        """Salva ``metrics.json`` com métricas + contexto do run.

        Args:
            metrics: Métricas a nível de vídeo (macro_f1, average_precision, ...).
            config: Snapshot resolvido da config Hydra (dict).
            threshold: Limiar calibrado (agregação RF ou sigmoid neural).
            aggregation_method: Método janela→vídeo (RF) ou "video_logit" (neural).
        """
        payload = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "metrics": {
                **metrics,
                "threshold": float(threshold),
                "aggregation_method": aggregation_method,
            },
            "config": config,
        }
        path = self.output_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
        log.info(f"Métricas salvas: {path}")
        return path

    def save_results_txt(self, metrics: dict[str, Any]) -> Path:
        """Salva um relatório legível em ``results.txt``."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("RELATÓRIO — BAH A/H Challenge (áudio + texto) — nível de vídeo")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Macro-F1 (oficial) : {metrics.get('macro_f1', float('nan')):.4f}")
        lines.append(f"Average Precision  : {metrics.get('average_precision', float('nan')):.4f}")
        if "n_videos" in metrics:
            lines.append(f"Vídeos avaliados   : {metrics['n_videos']}")
        if "threshold" in metrics:
            lines.append(f"Limiar calibrado   : {metrics['threshold']:.4f}")
        if "aggregation_method" in metrics:
            lines.append(f"Agregação          : {metrics['aggregation_method']}")

        if metrics.get("per_class"):
            lines.append("")
            lines.append("-" * 70)
            lines.append("Métricas por classe:")
            for cls_name, vals in metrics["per_class"].items():
                lines.append(f"  {cls_name}: {vals}")

        if metrics.get("classification_report"):
            lines.append("")
            lines.append("-" * 70)
            lines.append("classification_report (sklearn):")
            lines.append(str(metrics["classification_report"]))

        lines.append("")
        lines.append("=" * 70)
        path = self.output_dir / "results.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"Relatório salvo: {path}")
        return path

    # ==========================================================================
    # Plots
    # ==========================================================================

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
    ) -> Path | None:
        """Matriz de confusão a nível de vídeo (0 = sem A/H, 1 = com A/H)."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
        except ImportError as exc:  # pragma: no cover
            log.warning(f"Plot pulado (deps ausentes): {exc}")
            return None

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if normalize:
            cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        ax.set_title("Matriz de Confusão (nível de vídeo)")
        fig.tight_layout()
        path = self.plots_dir / "confusion_matrix_video.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Plot salvo: {path}")
        return path

    def plot_feature_importances(
        self,
        importances: np.ndarray | None,
        feature_names: list[str],
        top_k: int = 30,
    ) -> Path | None:
        """Top-k importâncias de features do RandomForest (barh).

        Para ``cross_attention`` não há importâncias → ``importances=None`` → pula.
        """
        if importances is None:
            log.warning("Sem feature_importances (modelo neural?); plot pulado.")
            return None
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            log.warning(f"Plot pulado (deps ausentes): {exc}")
            return None

        importances = np.asarray(importances, dtype=float)
        order = np.argsort(importances)[::-1][:top_k]
        names = [feature_names[i] for i in order]
        vals = importances[order]

        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(order))))
        ax.barh(range(len(order)), vals[::-1], color="#4C72B0")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("Importância (Gini)")
        ax.set_title(f"Top-{len(order)} Importâncias de Features (RandomForest)")
        fig.tight_layout()
        path = self.plots_dir / "feature_importances.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Plot salvo: {path}")
        return path

    def plot_grouped_feature_importances(
        self,
        importances: np.ndarray | None,
        group_of: list[str],
        top_k: int = 40,
    ) -> Path | None:
        """Importâncias **somadas por grupo** de features (barh, mesmo estilo).

        As importâncias (Gini) somam 1 sobre todas as features; somá-las por grupo dá a
        contribuição total do grupo. Útil p/ o panorama: tratar **texto** e **áudio** como
        1 feature cada e compará-los contra as features tabulares do dataset. Os grupos de
        embedding (rótulo contendo ``"emb"``) são destacados em cor diferente.

        Args:
            importances: vetor por feature (alinhado a ``group_of``).
            group_of: rótulo do grupo de cada feature (mesmo tamanho de ``importances``).
            top_k: máximo de grupos exibidos.
        """
        if importances is None:
            log.warning("Sem feature_importances (modelo neural?); plot agrupado pulado.")
            return None
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            log.warning(f"Plot pulado (deps ausentes): {exc}")
            return None

        importances = np.asarray(importances, dtype=float)
        if len(group_of) != len(importances):
            log.warning("group_of e importances de tamanhos diferentes; plot agrupado pulado.")
            return None

        agg: dict[str, float] = {}
        for imp, g in zip(importances, group_of, strict=False):
            agg[g] = agg.get(g, 0.0) + float(imp)
        labels = list(agg.keys())
        vals = np.array([agg[g] for g in labels], dtype=float)
        order = np.argsort(vals)[::-1][:top_k]
        names = [labels[i] for i in order]
        vals = vals[order]
        # Destaca os grupos de embedding (rótulo contém "emb") vs features tabulares.
        colors = ["#DD8452" if "emb" in n else "#4C72B0" for n in names]

        fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(order))))
        ax.barh(range(len(order)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_xlabel("Importância (Gini) somada por grupo")
        ax.set_title("Importância por grupo — embeddings vs features do dataset")
        fig.tight_layout()
        path = self.plots_dir / "feature_importances_grouped.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Plot salvo: {path}")
        return path

    def plot_precision_recall(self, y_true: np.ndarray, y_proba: np.ndarray) -> Path | None:
        """Curva Precision–Recall + Average Precision (classe positiva A/H)."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import average_precision_score, precision_recall_curve
        except ImportError as exc:  # pragma: no cover
            log.warning(f"Plot pulado (deps ausentes): {exc}")
            return None

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall, precision, color="#C44E52", lw=2, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(0.0, 1.0)
        ax.set_title("Curva Precision–Recall (classe A/H, nível de vídeo)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        path = self.plots_dir / "precision_recall.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Plot salvo: {path} (AP={ap:.3f})")
        return path

    def plot_threshold_curve(
        self,
        thresholds: np.ndarray,
        scores: np.ndarray,
        best_threshold: float,
    ) -> Path | None:
        """Curva limiar × Macro-F1 da calibração (``training.aggregation`` / sigmoid)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            log.warning(f"Plot pulado (deps ausentes): {exc}")
            return None

        thresholds = np.asarray(thresholds, dtype=float)
        scores = np.asarray(scores, dtype=float)
        best_idx = int(np.argmin(np.abs(thresholds - best_threshold)))
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(thresholds, scores, color="#55A868", lw=2)
        ax.axvline(
            best_threshold,
            color="#4C72B0",
            ls="--",
            lw=1.5,
            label=f"limiar* = {best_threshold:.3f}",
        )
        ax.scatter([best_threshold], [scores[best_idx]], color="#4C72B0", zorder=5)
        ax.set_xlabel("Limiar de decisão")
        ax.set_ylabel("Macro-F1 (validação, nível de vídeo)")
        ax.set_title("Calibração do limiar × Macro-F1")
        ax.legend(loc="lower center")
        fig.tight_layout()
        path = self.plots_dir / "threshold_calibration.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info(f"Plot salvo: {path}")
        return path

    # ==========================================================================
    # Export de predições e análise de erros (nível de vídeo)
    # ==========================================================================

    def save_predictions_csv(
        self,
        video_ids: np.ndarray,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> Path:
        """Salva ``predictions.csv`` com TP/TN/FP/FN e metadados por vídeo."""
        import csv

        metadata = metadata or {}
        path = self.output_dir / "predictions.csv"
        fieldnames = [
            "video_id",
            "y_true",
            "y_proba",
            "y_pred",
            "error_type",
            "margin",
            "participant_id",
            "question_type",
            "split",
            "ah_duration_s",
            "certainty_ah_mean",
        ]
        rows: list[dict[str, Any]] = []
        for i, vid in enumerate(video_ids):
            vid = str(vid)
            yt, yp, ypr = int(y_true[i]), float(y_proba[i]), int(y_pred[i])
            if yt == 1 and ypr == 1:
                err = "TP"
            elif yt == 0 and ypr == 0:
                err = "TN"
            elif yt == 0 and ypr == 1:
                err = "FP"
            else:
                err = "FN"
            meta = metadata.get(vid, {})
            rows.append(
                {
                    "video_id": vid,
                    "y_true": yt,
                    "y_proba": round(yp, 6),
                    "y_pred": ypr,
                    "error_type": err,
                    "margin": round(abs(yp - threshold), 6),
                    "participant_id": meta.get("participant_id", ""),
                    "question_type": meta.get("question_type", ""),
                    "split": meta.get("split", ""),
                    "ah_duration_s": meta.get("ah_duration_s", ""),
                    "certainty_ah_mean": meta.get("certainty_ah_mean", ""),
                }
            )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Predições salvas: {path} ({len(rows)} vídeos)")
        return path

    def save_error_analysis(
        self,
        video_ids: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> Path:
        """Agrega erros por ``question_type`` e ``participant_id`` em JSON."""
        from collections import defaultdict

        from src.training.metrics import per_class_f1, video_macro_f1

        metadata = metadata or {}
        by_qtype: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "y_true": [], "y_pred": []}
        )
        by_participant: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "y_true": [], "y_pred": []}
        )

        def _bump(bucket: dict, yt: int, ypr: int) -> None:
            bucket["n"] += 1
            bucket["y_true"].append(yt)
            bucket["y_pred"].append(ypr)
            if yt == 1 and ypr == 1:
                bucket["tp"] += 1
            elif yt == 0 and ypr == 0:
                bucket["tn"] += 1
            elif yt == 0 and ypr == 1:
                bucket["fp"] += 1
            else:
                bucket["fn"] += 1

        for i, vid in enumerate(video_ids):
            meta = metadata.get(str(vid), {})
            yt, ypr = int(y_true[i]), int(y_pred[i])
            qtype = meta.get("question_type") or "unknown"
            pid = meta.get("participant_id") or "unknown"
            _bump(by_qtype[qtype], yt, ypr)
            _bump(by_participant[pid], yt, ypr)

        def _summarize(groups: dict[str, dict]) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for key, g in sorted(groups.items()):
                yt = np.array(g["y_true"], dtype=np.int64)
                yp = np.array(g["y_pred"], dtype=np.int64)
                out.append(
                    {
                        "group": key,
                        "n_videos": g["n"],
                        "tp": g["tp"],
                        "tn": g["tn"],
                        "fp": g["fp"],
                        "fn": g["fn"],
                        "macro_f1": video_macro_f1(yt, yp),
                        "per_class": per_class_f1(yt, yp),
                    }
                )
            return out

        payload = {
            "by_question_type": _summarize(by_qtype),
            "by_participant": _summarize(by_participant),
        }
        path = self.output_dir / "errors_by_question_type.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
        log.info(f"Análise de erros salva: {path}")
        return path


def load_video_metadata_for_eval(
    cfg: Any,
    video_ids: list[str] | np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Metadados por vídeo (Parquet + anotações YAML) para relatórios de erro."""
    from pathlib import Path

    import polars as pl

    from src.data.indexing import load_annotation_yaml

    ids = {str(v) for v in video_ids}
    parquet_path = Path(cfg.data.paths.parquet_path)
    raw_root = Path(cfg.data.paths.data_root)
    ann_path = raw_root / "video_annotation_transcript.yaml"
    annotations = load_annotation_yaml(ann_path)

    df = pl.read_parquet(parquet_path)
    df = df.filter(pl.col("id").is_in(list(ids)))
    video_meta = (
        df.group_by("id")
        .agg(
            pl.col("participant_id").first(),
            pl.col("question_type").first(),
            pl.col("split").first(),
            pl.col("video_label").first(),
        )
        .to_dicts()
    )

    out: dict[str, dict[str, Any]] = {}
    for row in video_meta:
        vid = str(row["id"])
        ann = annotations.get(vid, {})
        intervals = ann.get("time_detailed_ah") or []
        ah_duration = 0.0
        for iv in intervals:
            if isinstance(iv, (list, tuple)) and len(iv) >= 2:
                ah_duration += float(iv[1]) - float(iv[0])
        cert = ann.get("certainty_ah") or []
        cert_mean = float(sum(cert) / len(cert)) if cert else None
        out[vid] = {
            "participant_id": row.get("participant_id"),
            "question_type": row.get("question_type"),
            "split": row.get("split"),
            "video_label": row.get("video_label"),
            "ah_duration_s": round(ah_duration, 3) if ah_duration else 0.0,
            "certainty_ah_mean": round(cert_mean, 3) if cert_mean is not None else "",
        }
    return out


def _json_default(obj: Any) -> Any:
    """Serializador de fallback p/ numpy no ``json.dump``."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
