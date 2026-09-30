"""ROI anatômico sobre MediaPipe Face Landmarker (478 pts).

Seleciona boca, olhos, sobrancelhas e íris — regiões mais relevantes para
sinais de ambivalência/hesitação facial, reduzindo ruído dos demais keypoints.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

RoiName = Literal["full", "ah"]

# Contornos / regiões MediaPipe Face Mesh (compatíveis com Face Landmarker 0–467).
_LIPS = (
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
)
_LEFT_EYE = (
    7,
    33,
    133,
    144,
    145,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    163,
    173,
    246,
)
_RIGHT_EYE = (
    249,
    263,
    362,
    373,
    374,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    390,
    398,
    466,
)
_LEFT_BROW = (46, 52, 53, 55, 63, 65, 66, 70, 105, 107)
_RIGHT_BROW = (276, 282, 283, 285, 293, 295, 296, 300, 334, 336)
# Face Landmarker Tasks: iris / pupil refinements (468–477).
_IRIS = tuple(range(468, 478))

_AH_INDICES = tuple(sorted(set(_LIPS + _LEFT_EYE + _RIGHT_EYE + _LEFT_BROW + _RIGHT_BROW + _IRIS)))

ROI_INDICES: dict[str, tuple[int, ...] | None] = {
    "full": None,
    "ah": _AH_INDICES,
}


def resolve_roi_indices(roi: str | None) -> np.ndarray | None:
    """Retorna índices ``(K,)`` ou ``None`` para malha completa."""
    key = (roi or "full").strip().lower()
    if key not in ROI_INDICES:
        raise ValueError(f"roi desconhecido: {roi!r} (use {sorted(ROI_INDICES)})")
    idxs = ROI_INDICES[key]
    if idxs is None:
        return None
    return np.asarray(idxs, dtype=np.int64)


def n_roi_landmarks(roi: str | None) -> int | None:
    """Número de landmarks após ROI, ou ``None`` se full (usa NUM_FACE_LANDMARKS)."""
    idxs = resolve_roi_indices(roi)
    return None if idxs is None else int(idxs.shape[0])
