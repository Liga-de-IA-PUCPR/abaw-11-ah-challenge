"""MediaPipe Face Landmarker — landmarks por janela temporal.

Usa a API ``mediapipe.tasks`` (0.10+). O bundle oficial produz **478** pontos
(mesh + íris). Janelas sem rosto detectável recebem zeros.

O ``extract`` agrupa por ``video_id`` e reutiliza um único ``VideoCapture`` por
arquivo — crítico p/ ~15k janelas.
"""

from __future__ import annotations

import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.schema import WindowSample
from src.logger import get_logger

log = get_logger("features.face_mesh")

# Face Landmarker (Tasks) → 478 landmarks (x, y, z).
NUM_FACE_LANDMARKS = 478
LANDMARK_DIM = 3

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_MODEL_PATH = Path("data/interim/models/face_landmarker.task")


def ensure_face_landmarker_model(model_path: str | Path | None = None) -> Path:
    """Garante que o ``.task`` do Face Landmarker exista (baixa sob demanda)."""
    path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Baixando Face Landmarker model → {path}")
    urllib.request.urlretrieve(_MODEL_URL, path)  # noqa: S310 — URL fixa do Google
    return path


class FaceMeshExtractor:
    """Extrator lazy de Face Landmarker (MediaPipe Tasks) sobre segmentos de vídeo."""

    def __init__(
        self,
        *,
        max_frames: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
        model_path: str | Path | None = None,
    ) -> None:
        self.max_frames = max(1, int(max_frames))
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.refine_landmarks = bool(refine_landmarks)
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._landmarker = None

    @property
    def dim(self) -> int:
        return NUM_FACE_LANDMARKS * LANDMARK_DIM

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for i in range(NUM_FACE_LANDMARKS):
            names.extend([f"face_{i}_x", f"face_{i}_y", f"face_{i}_z"])
        return names

    def _ensure_landmarker(self) -> None:
        if self._landmarker is not None:
            return
        import mediapipe as mp

        model = ensure_face_landmarker_model(self.model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _landmarks_from_frame(self, rgb: np.ndarray) -> np.ndarray | None:
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        pts = np.array(
            [[p.x, p.y, p.z] for p in result.face_landmarks[0]],
            dtype=np.float64,
        )
        if pts.shape[0] < NUM_FACE_LANDMARKS:
            return None
        return pts[:NUM_FACE_LANDMARKS]

    def extract_window(
        self,
        video_path: str | Path,
        t0: float,
        t1: float,
    ) -> np.ndarray:
        """Retorna landmarks ``(478, 3)`` médios sobre frames amostrados em ``[t0, t1]``."""
        video_path = Path(video_path)
        if not video_path.exists():
            return np.zeros((NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float32)

        import cv2

        self._ensure_landmarker()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return np.zeros((NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        try:
            return self._extract_from_cap(cap, t0, t1)
        finally:
            cap.release()

    def _extract_from_cap(self, cap, t0: float, t1: float) -> np.ndarray:
        import cv2

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
            pts = self._landmarks_from_frame(rgb)
            if pts is None:
                continue
            accum += pts
            count += 1
        if count == 0:
            return np.zeros((NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        return (accum / count).astype(np.float32)

    def extract(
        self,
        windows: list[WindowSample],
        video_root: str | Path,
    ) -> np.ndarray:
        """Extrai landmarks para um lote de janelas → ``(N, 478, 3)``.

        Agrupa por ``video_id`` e reabre o mp4 só uma vez por vídeo no lote.
        """
        import cv2

        video_root = Path(video_root)
        self._ensure_landmarker()
        out = np.zeros((len(windows), NUM_FACE_LANDMARKS, LANDMARK_DIM), dtype=np.float32)

        by_video: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(windows):
            by_video[w.video_id].append(i)

        missing = 0
        for video_id, idxs in by_video.items():
            mp4 = video_root / video_id
            if not mp4.exists():
                missing += len(idxs)
                continue
            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened():
                missing += len(idxs)
                continue
            try:
                for i in idxs:
                    w = windows[i]
                    out[i] = self._extract_from_cap(cap, w.t0, w.t1)
            finally:
                cap.release()

        if missing:
            log.warning(f"{missing}/{len(windows)} janelas sem mp4 em {video_root}")
        return out

    def flatten(self, landmarks: np.ndarray) -> np.ndarray:
        """``(N, 478, 3)`` → ``(N, 1434)`` para gravação no Parquet."""
        if landmarks.ndim == 2:
            return landmarks.astype(np.float32)
        return landmarks.reshape(landmarks.shape[0], -1).astype(np.float32)
