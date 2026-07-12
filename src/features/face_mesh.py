"""MediaPipe Face Mesh — 468 landmarks por janela temporal.

Extrai keypoints faciais alinhados a ``[t0, t1]`` de cada ``WindowSample``.
O resultado é um array ``(468, 3)`` com coordenadas normalizadas (x, y, z) em
espaço de imagem MediaPipe. Janelas sem rosto detectável recebem zeros.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.schema import WindowSample
from src.logger import get_logger

log = get_logger("features.face_mesh")

NUM_FACE_LANDMARKS = 468
LANDMARK_DIM = 3


class FaceMeshExtractor:
    """Extrator lazy de Face Mesh (MediaPipe) sobre segmentos de vídeo."""

    def __init__(
        self,
        *,
        max_frames: int = 8,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        self.max_frames = max(1, int(max_frames))
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.refine_landmarks = bool(refine_landmarks)
        self._mesh = None

    @property
    def dim(self) -> int:
        return NUM_FACE_LANDMARKS * LANDMARK_DIM

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for i in range(NUM_FACE_LANDMARKS):
            names.extend([f"face_{i}_x", f"face_{i}_y", f"face_{i}_z"])
        return names

    def _ensure_mesh(self) -> None:
        if self._mesh is not None:
            return
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None

    def extract_window(
        self,
        video_path: str | Path,
        t0: float,
        t1: float,
    ) -> np.ndarray:
        """Retorna landmarks ``(468, 3)`` médios sobre frames amostrados em ``[t0, t1]``."""
        video_path = Path(video_path)
        if not video_path.exists():
            return np.zeros((NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float32)

        import cv2

        self._ensure_mesh()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return np.zeros((NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float32)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        duration = max(float(t1) - float(t0), 1e-3)
        n_samples = min(self.max_frames, max(1, int(duration * fps)))
        times = np.linspace(float(t0), float(t1), num=n_samples, endpoint=False)

        accum = np.zeros((NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float64)
        count = 0
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue
            lm = result.multi_face_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)
            if pts.shape[0] != NUM_FACE_LANDMARKS:
                continue
            accum += pts
            count += 1

        cap.release()
        if count == 0:
            return np.zeros((NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        return (accum / count).astype(np.float32)

    def extract(
        self,
        windows: list[WindowSample],
        video_root: str | Path,
    ) -> np.ndarray:
        """Extrai landmarks para um lote de janelas → ``(N, 468, 3)``."""
        video_root = Path(video_root)
        out = np.zeros((len(windows), NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        missing = 0
        for i, w in enumerate(windows):
            mp4 = video_root / w.video_id
            if not mp4.exists():
                missing += 1
                continue
            out[i] = self.extract_window(mp4, w.t0, w.t1)
        if missing:
            log.warning(f"{missing}/{len(windows)} janelas sem mp4 em {video_root}")
        return out

    def flatten(self, landmarks: np.ndarray) -> np.ndarray:
        """``(N, 468, 3)`` → ``(N, 1404)`` para gravação no Parquet."""
        if landmarks.ndim == 2:
            return landmarks.astype(np.float32)
        return landmarks.reshape(landmarks.shape[0], -1).astype(np.float32)
