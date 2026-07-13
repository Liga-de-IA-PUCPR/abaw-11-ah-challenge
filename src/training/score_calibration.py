"""Calibração de scores contínuos a nível de vídeo (antes do limiar).

Temperature scaling: ajusta um parâmetro T minimizando NLL (BCE) na validação.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from src.logger import get_logger

log = get_logger("training.score_calibration")

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """``sigmoid(logit(s) / T)`` — scores calibrados em [0, 1]."""
    t = max(float(temperature), _EPS)
    return _sigmoid(_logit(scores) / t).astype(np.float32)


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    t_min: float = 0.05,
    t_max: float = 5.0,
) -> float:
    """Otimiza T minimizando NLL binária (cross-entropy) na validação."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if len(scores) == 0:
        log.warning("fit_temperature: sem amostras; T=1.0")
        return 1.0

    def nll(t: float) -> float:
        p = apply_temperature(scores, t).astype(np.float64)
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    result = minimize_scalar(nll, bounds=(t_min, t_max), method="bounded")
    t_opt = float(result.x) if result.success else 1.0
    log.info(f"Temperature scaling: T={t_opt:.4f} (NLL val={nll(t_opt):.4f})")
    return t_opt


def serialize_calibrator(calibration_type: str, **params: Any) -> dict[str, Any]:
    return {"type": calibration_type, **params}


def apply_calibrator(scores: np.ndarray, calibrator: dict[str, Any] | None) -> np.ndarray:
    if not calibrator or calibrator.get("type") in (None, "none"):
        return np.asarray(scores, dtype=np.float32)
    if calibrator["type"] == "temperature":
        return apply_temperature(scores, float(calibrator["T"]))
    raise ValueError(f"Calibrador desconhecido: {calibrator.get('type')}")
