"""Métricas a nível de vídeo do desafio BAH (sklearn, CPU — definição canônica).

- ``video_macro_f1``: F1 macro (classes 0 e 1) — métrica oficial.
- ``average_precision``: AP da classe positiva (A/H = 1).
- ``per_class_f1`` / ``confusion``: diagnóstico por classe.

Não importa torch. No caminho Lightning usa-se torchmetrics (ver nota da FASE 4).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from src.logger import get_logger

log = get_logger("training.metrics")


def video_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 a nível de vídeo (métrica oficial do desafio)."""
    return float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision da classe positiva (A/H = 1); 0.0 se houver só uma classe."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        log.warning("AP indefinida (uma única classe verdadeira); retornando 0.0")
        return 0.0
    return float(average_precision_score(y_true, y_score))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """F1 por classe (0 = sem A/H, 1 = com A/H)."""
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    return {"f1_class_0": float(f1s[0]), "f1_class_1": float(f1s[1])}


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Matriz de confusão 2x2 (linhas = verdadeiro, colunas = predito)."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def evaluate_video_predictions(
    video_labels: dict[str, int],
    video_pred: dict[str, int],
    video_score: dict[str, float] | None = None,
) -> dict[str, object]:
    """Avalia predições a nível de vídeo (alinha pelos ``video_id`` em comum).

    Returns:
        Dict com ``macro_f1``, ``per_class``, ``confusion_matrix``,
        ``average_precision`` (se ``video_score``) e ``n_videos``.
    """
    ids = [v for v in video_labels if v in video_pred]
    y_true = np.array([video_labels[v] for v in ids], dtype=np.int64)
    y_pred = np.array([video_pred[v] for v in ids], dtype=np.int64)

    report: dict[str, object] = {
        "n_videos": len(ids),
        "macro_f1": video_macro_f1(y_true, y_pred),
        "per_class": per_class_f1(y_true, y_pred),
        "confusion_matrix": confusion(y_true, y_pred).tolist(),
    }
    if video_score is not None:
        y_score = np.array([video_score.get(v, 0.0) for v in ids], dtype=np.float32)
        report["average_precision"] = average_precision(y_true, y_score)

    log.info(
        f"Avaliação (vídeo): macro_f1={report['macro_f1']:.4f} sobre {report['n_videos']} vídeos"
    )
    return report
