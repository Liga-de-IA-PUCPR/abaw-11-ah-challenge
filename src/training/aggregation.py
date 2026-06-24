"""Agregação janela → vídeo + calibração de limiar (contrato §6.5).

RF: cada vídeo tem várias janelas com proba positiva ->
  1. agrega num score por vídeo, 2. aplica limiar calibrado na val (max macro-F1).
Cross-attention: já tem 1 score (sigmoid) por vídeo -> ``method="identity"``.

Métodos: "mean_proba" (default), "max_proba", "frac_positive", "any", "identity".
"""

from __future__ import annotations

import numpy as np

from src.logger import get_logger
from src.training.metrics import video_macro_f1

log = get_logger("training.aggregation")

VALID_METHODS = ("mean_proba", "max_proba", "frac_positive", "any", "identity")


def _video_scores(
    window_proba: np.ndarray, window_video_ids: np.ndarray, method: str
) -> dict[str, float]:
    """Agrega probas de janela num score por vídeo (independe do limiar).

    Para "any" o score é o **máximo** (limiar aplicado a ele = "existe janela >=
    limiar"). Para "identity" espera-se 1 score por vídeo (cross-attention) —
    usa o máximo como redução trivial caso haja duplicatas.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Método inválido: '{method}'. Use {VALID_METHODS}")

    window_proba = np.asarray(window_proba, dtype=np.float32)
    window_video_ids = np.asarray(window_video_ids)

    scores: dict[str, float] = {}
    for vid in np.unique(window_video_ids):
        p = window_proba[window_video_ids == vid]
        if method == "mean_proba":
            score = float(p.mean())
        elif method in ("max_proba", "any", "identity"):
            score = float(p.max())
        elif method == "frac_positive":
            score = float((p >= 0.5).mean())
        scores[str(vid)] = score
    return scores


def video_scores(
    window_proba: np.ndarray, window_video_ids: np.ndarray, method: str
) -> dict[str, float]:
    """API pública: score contínuo por vídeo (para AP e relatórios)."""
    return _video_scores(window_proba, window_video_ids, method)


def aggregate_to_video(
    window_proba: np.ndarray,
    window_video_ids: np.ndarray,
    method: str,
    threshold: float,
) -> dict[str, int]:
    """Agrega janelas em uma predição binária por vídeo (§6.5).

    Returns:
        ``{video_id: pred 0/1}``.
    """
    scores = _video_scores(window_proba, window_video_ids, method)
    return {vid: int(score >= threshold) for vid, score in scores.items()}


def calibrate_threshold(
    val_proba: np.ndarray,
    val_video_ids: np.ndarray,
    val_video_labels: dict[str, int],
    method: str,
    metric: str = "macro_f1",
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Varre um grid de limiares em [0, 1] maximizando o Macro-F1 na validação.

    Os scores por vídeo são calculados **uma vez**; só a binarização varia ao
    longo do grid (busca barata). Serve tanto ao RF (probas de janela) quanto à
    cross-attention (``method="identity"``: 1 sigmoid por vídeo).

    Returns:
        ``(melhor_limiar, melhor_macro_f1)``.
    """
    if metric != "macro_f1":
        raise ValueError(f"Métrica de calibração não suportada: '{metric}'")
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    scores = _video_scores(val_proba, val_video_ids, method)
    ids = [v for v in scores if v in val_video_labels]
    if not ids:
        log.warning("Sem vídeos rotulados para calibrar; usando limiar 0.5")
        return 0.5, 0.0

    y_true = np.array([val_video_labels[v] for v in ids], dtype=np.int64)
    s = np.array([scores[v] for v in ids], dtype=np.float32)

    best_thr, best_score = 0.5, -1.0
    for thr in grid:
        f1 = video_macro_f1(y_true, (s >= thr).astype(np.int64))
        if f1 > best_score:
            best_score, best_thr = f1, float(thr)

    log.info(f"Limiar calibrado (method='{method}'): thr={best_thr:.3f} -> macro_f1={best_score:.4f}")
    return best_thr, best_score
