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


def _moving_average(y: np.ndarray, k: int) -> np.ndarray:
    """Média móvel centrada; nas bordas encolhe a janela (sem viés de zero-padding)."""
    y = np.asarray(y, dtype=np.float64)
    if k <= 1:
        return y
    half, n = k // 2, len(y)
    out = np.empty_like(y)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


# Tolerância (em macro-F1) que define o platô "quase-máximo" na seleção 'smooth':
# limiares cujo F1 suavizado está a <= _PLATEAU_TOL do máximo contam como platô. Numa
# val pequena (~124 vídeos) diferenças dessa ordem são ruído — então escolhemos o CENTRO
# da faixa indistinguível do máximo, não uma borda dela.
_PLATEAU_TOL = 0.01


def _select_threshold_index(
    grid: np.ndarray, f1s: np.ndarray, selection: str, smooth_window: float
) -> int:
    """Índice do limiar escolhido na curva F1×limiar, conforme a estratégia.

    - ``"argmax"``: pico cru — sensível a ruído quando a val é pequena (a curva é
      uma função degrau e um spike de 1-2 vídeos pode vencer mas não generalizar).
    - ``"smooth"`` (default): suaviza a curva (média móvel de largura ``smooth_window``)
      e escolhe o **CENTRO do maior platô** cujo F1 suavizado está a ``_PLATEAU_TOL`` do
      máximo. Diferente do ``argmax`` da curva suavizada (que numa val plana escorrega
      para uma borda — ex.: 0.5), o centro do platô transfere melhor para o test/hidden-test.
    """
    if selection == "argmax":
        return int(np.argmax(f1s))
    if selection == "smooth":
        spacing = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
        k = max(1, int(round(smooth_window / spacing)))
        s = _moving_average(f1s, k)
        plateau = np.flatnonzero(s >= float(s.max()) - _PLATEAU_TOL)
        if plateau.size == 0:
            return int(np.argmax(s))
        runs = np.split(plateau, np.flatnonzero(np.diff(plateau) > 1) + 1)
        best = max(runs, key=len)
        return int(best[len(best) // 2])
    raise ValueError(f"selection inválida: '{selection}'. Use 'smooth' | 'argmax'.")


def _calibrate_base_rate(
    s: np.ndarray,
    y_true: np.ndarray,
    target_pos_rate: float | None,
    method_label: str,
) -> tuple[float, float]:
    """Limiar por casamento de taxa-base (quantil dos scores)."""
    if len(np.unique(y_true)) < 2:
        log.warning(
            "Conjunto de calibração tem UMA ÚNICA classe (%d vídeos) — o limiar resultante "
            "não tem sentido.",
            len(y_true),
        )
    p_plus = float(y_true.mean()) if target_pos_rate is None else float(target_pos_rate)
    best_thr = float(np.quantile(s, 1.0 - p_plus))
    best_score = video_macro_f1(y_true, (s >= best_thr).astype(np.int64))
    log.info(
        f"Limiar calibrado ({method_label}, selection='base_rate', "
        f"p+={p_plus:.3f}): thr={best_thr:.3f} -> macro_f1={best_score:.4f}"
    )
    return best_thr, best_score


def calibrate_threshold(
    val_proba: np.ndarray,
    val_video_ids: np.ndarray,
    val_video_labels: dict[str, int],
    method: str,
    metric: str = "macro_f1",
    grid: np.ndarray | None = None,
    selection: str = "smooth",
    smooth_window: float = 0.10,
    target_pos_rate: float | None = None,
) -> tuple[float, float]:
    """Varre um grid de limiares em [0, 1] e escolhe o melhor na validação.

    A escolha do limiar usa ``selection``:
    - ``"argmax"``: pico cru da curva F1×limiar.
    - ``"smooth"``: centro do platô da curva suavizada (ver ``_select_threshold_index``).
    - ``"base_rate"``: casamento de taxa-base — limiar cujo quantil reproduz ``target_pos_rate``
      (fração predita positiva). ``target_pos_rate=None`` usa a prevalência na val.

    Returns:
        ``(melhor_limiar, macro_f1_no_limiar)``.
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

    if selection == "base_rate":
        return _calibrate_base_rate(
            s, y_true, target_pos_rate, f"method='{method}'"
        )

    f1s = np.array(
        [video_macro_f1(y_true, (s >= thr).astype(np.int64)) for thr in grid],
        dtype=np.float64,
    )
    best_idx = _select_threshold_index(grid, f1s, selection, smooth_window)
    best_thr, best_score = float(grid[best_idx]), float(f1s[best_idx])

    log.info(
        f"Limiar calibrado (method='{method}', selection='{selection}'): "
        f"thr={best_thr:.3f} -> macro_f1={best_score:.4f}"
    )
    return best_thr, best_score


def calibrate_threshold_from_video_scores(
    video_scores_arr: np.ndarray,
    video_labels: np.ndarray,
    metric: str = "macro_f1",
    grid: np.ndarray | None = None,
    selection: str = "smooth",
    smooth_window: float = 0.10,
    target_pos_rate: float | None = None,
) -> tuple[float, float]:
    """Varre limiares sobre scores **já agregados** por vídeo (ex.: pós temperature scaling)."""
    if metric != "macro_f1":
        raise ValueError(f"Métrica de calibração não suportada: '{metric}'")
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    y_true = np.asarray(video_labels, dtype=np.int64)
    s = np.asarray(video_scores_arr, dtype=np.float32)
    if len(s) == 0:
        log.warning("Sem scores para calibrar; usando limiar 0.5")
        return 0.5, 0.0

    if selection == "base_rate":
        return _calibrate_base_rate(
            s, y_true, target_pos_rate, "scores pré-agregados"
        )

    f1s = np.array(
        [video_macro_f1(y_true, (s >= thr).astype(np.int64)) for thr in grid],
        dtype=np.float64,
    )
    best_idx = _select_threshold_index(grid, f1s, selection, smooth_window)
    best_thr, best_score = float(grid[best_idx]), float(f1s[best_idx])
    log.info(
        f"Limiar calibrado (scores pré-agregados, selection='{selection}'): "
        f"thr={best_thr:.3f} -> macro_f1={best_score:.4f}"
    )
    return best_thr, best_score


def threshold_curve(
    y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Curva limiar × Macro-F1 sobre scores **a nível de vídeo** (para o Reporter, FASE 5)."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)
    f1s = np.array(
        [video_macro_f1(y_true, (y_score >= t).astype(np.int64)) for t in grid],
        dtype=np.float32,
    )
    return grid, f1s
