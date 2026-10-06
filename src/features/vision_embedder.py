"""Embedders visuais do plano MoE — recortes de rosto (por janela) e cena (por vídeo).

Mesmo contrato dos embedders de texto/áudio (:class:`~src.base.embedder.BaseEmbedder`) e
mesma filosofia: **agnósticos ao modelo** — qualquer ``AutoModel`` de visão do HuggingFace
serve (``model_name`` no YAML). Nenhuma dependência nova: só ``transformers`` + ``PIL``
(+ ``opencv`` do grupo ``vision`` p/ ler o fps/frames dos ``.mp4``).

- :class:`HFImageEmbedder` — imagens → ``(n, dim)`` (CLS | média dos tokens | pooler).
- :class:`FaceCropEmbedder` — canal FACIAL da figura do plano ("recortes face / olhos / boca
  → backbone"). Usa os rostos JÁ recortados e alinhados que o BAH distribui
  (``cropped-aligned-faces/<video_id>/frame-<n>.jpg``, 256×256): como o alinhamento põe olhos
  e boca em posições canônicas, os recortes de olhos/boca são caixas fixas (frações da
  imagem, configuráveis) — a mesma ideia de ``EyesMeshCrop``/``MouthMeshCrop`` da
  ``vision-toolbelt-liga``, sem rodar detector/Face Mesh por frame. Amostra frames numa
  grade de ``sample_fps`` por vídeo (compartilhada pelas janelas sobrepostas), embeda cada
  recorte e grava, por janela, ``[face ‖ eyes ‖ mouth]`` (média dos frames da janela).
- :class:`SceneEmbedder` — canal de CENA opcional (VideoMAE congelado sobre ``num_frames``
  frames do vídeo inteiro, não recortado) → vetor por vídeo; também carrega o VideoMAE-v2
  (código remoto, ``scene_embedder=videomae_v2``).

A dinâmica temporal (Transformer + ``[μ, σ, μΔ, σΔ]``) e a mistura dos recortes (SoftMoE)
ficam no MODELO (``src/models/encoders.py``), não aqui: o cache guarda só o que é caro.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.base.embedder import BaseEmbedder
from src.conf import resolve_device
from src.logger import get_logger

log = get_logger("features.vision_embedder")

# Caixas (x0, y0, x1, y1) em fração da imagem ALINHADA do BAH (256×256): olhos ~45% da
# altura (com sobrancelhas), boca ~82%. Sobreponha em vision_embedder.crops no YAML.
DEFAULT_CROPS: dict[str, Sequence[float]] = {
    "face": (0.0, 0.0, 1.0, 1.0),
    "eyes": (0.08, 0.26, 0.92, 0.60),
    "mouth": (0.24, 0.66, 0.76, 0.98),
}


def _pool_hidden(out, pooling: str):
    """Vetor por item a partir da saída de um ``AutoModel`` de visão (ViT/ConvNet/VideoMAE).

    Modelos de código remoto que já devolvem o vetor (ex.: VideoMAE-v2) passam direto.
    """
    if not hasattr(out, "last_hidden_state"):
        return out if out.dim() == 2 else out.mean(dim=1)
    if pooling == "pooler" and getattr(out, "pooler_output", None) is not None:
        return out.pooler_output.flatten(1)
    hidden = out.last_hidden_state
    if hidden.dim() == 4:  # ConvNet: (B, C, H, W) → média espacial
        return hidden.mean(dim=(2, 3))
    if pooling == "cls":
        return hidden[:, 0]
    return hidden.mean(dim=1)  # tokens (ViT/VideoMAE)


def _load_hf_vision(model_name: str, trust_remote_code: bool, device):
    """``(processor, modelo em eval no device)`` de um repositório de visão do HuggingFace."""
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    return processor, model.to(device).eval()


def _config_dim(model) -> int | None:
    """Dimensão declarada no config (ViT: ``hidden_size``; ConvNets: ``hidden_sizes[-1]``).

    ``None`` quando o config não declara (código remoto) — aí a dimensão sai de 1 forward.
    """
    config = model.config
    sizes = getattr(config, "hidden_sizes", None) or [None]
    return getattr(config, "hidden_size", None) or sizes[-1]


class HFImageEmbedder(BaseEmbedder):
    """Imagens → embedding via ``AutoImageProcessor`` + ``AutoModel`` (device-aware)."""

    name: str = "image"

    def __init__(
        self,
        model_name: str = "trpakov/vit-face-expression",
        pooling: str = "cls",
        batch_size: int = 64,
        trust_remote_code: bool = False,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)
        log.info(f"Carregando backbone visual: {model_name}")
        self.processor, self.model = _load_hf_vision(model_name, trust_remote_code, self.device)
        self._dim = _config_dim(self.model)

    @property
    def dim(self) -> int:
        if self._dim is None:  # config sem a dimensão: 1 forward com uma imagem em branco
            self._dim = self._embed([np.zeros((224, 224, 3), dtype=np.uint8)]).shape[1]
        return int(self._dim)

    def extract(self, inputs: list) -> np.ndarray:
        """Lista de imagens (``PIL.Image`` ou ``np.ndarray`` HxWx3) → ``(n, dim)`` float32."""
        if not inputs:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = [
            self._embed(inputs[start : start + self.batch_size])
            for start in range(0, len(inputs), self.batch_size)
        ]
        return self._validate_output(np.concatenate(out, axis=0), len(inputs))

    def _embed(self, images: list) -> np.ndarray:
        import torch

        with torch.no_grad():
            batch = self.processor(images=images, return_tensors="pt").to(self.device)
            return _pool_hidden(self.model(**batch), self.pooling).float().cpu().numpy()

    def feature_names(self) -> list[str]:
        return [f"image_emb_{i}" for i in range(self.dim)]


class FaceCropEmbedder:
    """Janelas de um vídeo → ``[face ‖ eyes ‖ mouth]`` por janela (média dos frames amostrados).

    Args:
        embedder: backbone de imagem (:class:`HFImageEmbedder` ou qualquer ``BaseEmbedder``).
        frames_root: raiz ``cropped-aligned-faces`` (``<frames_root>/<video_id>/frame-<n>.jpg``).
        crops: nome → caixa ``(x0, y0, x1, y1)`` em fração da imagem alinhada.
        sample_fps: frames amostrados por segundo de vídeo (grade compartilhada entre janelas).
        default_fps: fps assumido quando o ``.mp4`` não está disponível p/ consulta.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        frames_root: str | Path,
        crops: dict[str, Sequence[float]] | None = None,
        sample_fps: float = 1.0,
        default_fps: float = 30.0,
    ) -> None:
        self.embedder = embedder
        self.frames_root = Path(frames_root)
        self.crops = {k: tuple(float(x) for x in v) for k, v in (crops or DEFAULT_CROPS).items()}
        self.sample_fps = float(sample_fps)
        self.default_fps = float(default_fps)

    @property
    def dim(self) -> int:
        return len(self.crops) * self.embedder.dim

    def feature_names(self) -> list[str]:
        return [f"face_{c}_{i}" for c in self.crops for i in range(self.embedder.dim)]

    def embed_video(
        self, video_id: str, spans: list[tuple[float, float]], video_path: Path | None
    ) -> np.ndarray:
        """``(len(spans), dim)`` p/ as janelas ``spans`` de um vídeo (zeros sem rosto)."""
        from PIL import Image

        frame_dir = self.frames_root / video_id
        out = np.zeros((len(spans), self.dim), dtype=np.float32)
        if not spans or not frame_dir.is_dir():
            return out
        fps = video_fps(video_path, self.default_fps)
        end = max(t1 for _, t1 in spans)
        step = 1.0 / self.sample_fps
        times = np.arange(step / 2.0, end, step)  # centro de cada passo da grade
        found = [(t, _nearest_frame(frame_dir, round(t * fps))) for t in times]
        found = [(t, f) for t, f in found if f is not None]
        if not found:
            return out
        images = [Image.open(f).convert("RGB") for _, f in found]
        crops = [_crop(img, box) for img in images for box in self.crops.values()]
        emb = self.embedder.extract(crops).reshape(len(found), self.dim)  # (frames, C·D)
        frame_t = np.array([t for t, _ in found])
        for i, (t0, t1) in enumerate(spans):
            sel = (frame_t >= t0) & (frame_t < t1)
            if sel.any():
                out[i] = emb[sel].mean(axis=0)
        return out


class SceneEmbedder(BaseEmbedder):
    """Vídeo inteiro → 1 vetor (VideoMAE congelado, ``num_frames`` frames uniformes).

    ``input_layout="bcthw"`` + ``trust_remote_code=True`` habilitam o VideoMAE-v2 do Hub
    (código remoto: entrada ``(B, C, T, H, W)``, saída já é o vetor do clipe); o default
    ``btchw`` é o layout do VideoMAE do transformers.
    """

    name: str = "scene"

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        num_frames: int = 16,
        pooling: str = "mean",
        input_layout: str = "btchw",
        trust_remote_code: bool = False,
        device: str = "auto",
    ) -> None:
        if input_layout not in ("btchw", "bcthw"):
            raise ValueError(f"input_layout desconhecido: {input_layout!r} (btchw|bcthw)")
        self.model_name = model_name
        self.num_frames = int(num_frames)
        self.pooling = pooling
        self.input_layout = input_layout
        self.device = resolve_device(device)
        log.info(f"Carregando encoder de cena: {model_name}")
        self.processor, self.model = _load_hf_vision(model_name, trust_remote_code, self.device)
        self._dim = _config_dim(self.model)

    @property
    def dim(self) -> int:
        if self._dim is None:  # config sem a dimensão: 1 forward com um clipe em branco
            blank = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            self._dim = self._embed(blank).shape[0]
        return int(self._dim)

    def extract(self, inputs: list) -> np.ndarray:
        """Lista de caminhos ``.mp4`` → ``(n, dim)`` (zeros p/ vídeo ilegível)."""
        out = np.zeros((len(inputs), self.dim), dtype=np.float32)
        for i, path in enumerate(inputs):
            frames = read_uniform_frames(path, self.num_frames)
            if frames is not None:
                out[i] = self._embed(frames)
        return self._validate_output(out, len(inputs))

    def _embed(self, frames: np.ndarray) -> np.ndarray:
        """Clipe ``(T, H, W, 3)`` → vetor ``(dim,)``."""
        import torch

        pixels = self.processor(list(frames), return_tensors="pt")["pixel_values"]
        if self.input_layout == "bcthw":  # (B, T, C, H, W) do processor → (B, C, T, H, W)
            pixels = pixels.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            out = self.model(pixel_values=pixels.to(self.device))
        return _pool_hidden(out, self.pooling)[0].float().cpu().numpy()

    def feature_names(self) -> list[str]:
        return [f"scene_emb_{i}" for i in range(self.dim)]


# ==============================================================================
# Helpers de vídeo/frames (OpenCV lazy — grupo opcional `vision`)
# ==============================================================================


def video_fps(video_path: Path | None, default: float) -> float:
    """fps do ``.mp4`` (metadado, sem decodificar); ``default`` se ausente/ilegível."""
    if video_path is None or not Path(video_path).exists():
        return default
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 0 else default


def read_uniform_frames(video_path: str | Path, num_frames: int) -> np.ndarray | None:
    """``num_frames`` frames RGB uniformes do vídeo inteiro → ``(T, H, W, 3)`` | ``None``."""
    if not Path(video_path).exists():
        return None
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return None
        frames = []
        for idx in np.linspace(0, total - 1, num_frames).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        return None
    while len(frames) < num_frames:  # completa com o último frame legível
        frames.append(frames[-1])
    return np.stack(frames)


def _nearest_frame(frame_dir: Path, n: int, radius: int = 3) -> Path | None:
    """``frame-<n>.jpg`` ou o vizinho mais próximo em ±``radius`` (frames sem rosto faltam)."""
    for d in [0] + [s * k for k in range(1, radius + 1) for s in (1, -1)]:
        f = frame_dir / f"frame-{n + d}.jpg"
        if n + d >= 0 and f.exists():
            return f
    return None


def _crop(img, box: Sequence[float]):
    """Recorte de ``img`` (PIL) pela caixa fracionária ``(x0, y0, x1, y1)``."""
    w, h = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
