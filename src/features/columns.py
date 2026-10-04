"""Colunas extras do Parquet de janelas — sinais novos SEM refazer o ``mode=featurize``.

O Parquet canônico (README §6.2) tem ``audio_emb``/``text_emb``/``tabular`` por janela. Os
sinais do plano MoE entram como COLUNAS ADICIONAIS no mesmo arquivo ("um Parquet, várias
visões"), cada uma produzida por um :class:`ColumnFeaturizer` registrado aqui e gravada por
``mode=featurize_columns columns=[...]`` (:mod:`src.pipeline.featurize_columns`):

=====================  ===========================  ===========================================
``columns=``           coluna                       conteúdo
=====================  ===========================  ===========================================
``transcript``         ``transcript``               transcrição completa (texto do fine-tune)
``asr_timing``         ``asr_timing``               16 gaps do Whisper (vídeo, replicado)
``hesitation_markers`` ``hesitation_markers``       11 marcadores léxicos (vídeo, replicado)
``audio``              ``audio_emb_<embedder>``     embedding por janela do ``audio_embedder``
``face_crops``         ``face_crops_<embedder>``    ``[face‖eyes‖mouth]`` por janela
``scene``              ``scene_emb_<embedder>``     VideoMAE do vídeo inteiro (replicado)
=====================  ===========================  ===========================================

Trocar o embedder de uma modalidade = gravar outra coluna (ex.: ``audio_embedder=hubert`` →
``audio_emb_hubert``) e apontar o ramo do modelo para ela (``model.branches.audio.column``).
Sinais de nível de vídeo são replicados nas janelas, como recomendado no plano original.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.data.schema import VideoRecord, WindowSample
from src.logger import get_logger

log = get_logger("features.columns")


class ColumnFeaturizer(ABC):
    """Produz UMA coluna do Parquet de janelas (1 valor por janela, na ordem de ``windows``).

    ``needs_records=True`` faz o pipeline montar o índice de vídeos (transcrição, chunks do
    Whisper, ``.mp4``) e passá-lo em ``records`` (``{video_id: VideoRecord}``).
    """

    needs_records: bool = False

    @property
    @abstractmethod
    def column(self) -> str:
        """Nome da coluna gravada no Parquet."""

    @abstractmethod
    def compute(
        self, windows: list[WindowSample], records: dict[str, VideoRecord] | None
    ) -> list[Any]:
        """Um valor por janela (``list[float]`` p/ vetores, ``str`` p/ texto)."""

    def feature_names(self) -> list[str]:
        """Nomes das dimensões (vão no sidecar JSON); vazio p/ colunas não vetoriais."""
        return []


class VideoLevelColumn(ColumnFeaturizer):
    """Valor calculado UMA vez por vídeo e replicado nas janelas dele."""

    needs_records = True

    @abstractmethod
    def compute_video(self, record: VideoRecord | None) -> Any:
        """Valor do vídeo (``record`` é ``None`` se o vídeo não estiver no índice)."""

    def compute(self, windows, records):
        records = records or {}
        cache: dict[str, Any] = {}
        for w in windows:
            if w.video_id not in cache:
                cache[w.video_id] = self.compute_video(records.get(w.video_id))
        return [cache[w.video_id] for w in windows]


class TranscriptColumn(VideoLevelColumn):
    """Transcrição completa do vídeo — entrada do ramo de texto fine-tunado (tokenizada no
    dataset, ver ``VideoSequenceDataset(transcript=...)``)."""

    column = "transcript"

    def compute_video(self, record):
        return (record.full_transcript if record is not None else "") or ""


class AsrTimingColumn(VideoLevelColumn):
    """16 features de "tempo apagado pelo ASR" (:mod:`src.features.asr_timing`)."""

    column = "asr_timing"

    def compute_video(self, record):
        from src.features.asr_timing import extract_asr_timing_vector

        return extract_asr_timing_vector(record.transcript_chunks if record else []).tolist()

    def feature_names(self):
        from src.features.asr_timing import ASR_TIMING_FEATURE_NAMES

        return list(ASR_TIMING_FEATURE_NAMES)


class HesitationMarkersColumn(VideoLevelColumn):
    """11 marcadores léxicos de hesitação (:mod:`src.features.hesitation_markers`)."""

    column = "hesitation_markers"

    def compute_video(self, record):
        from src.features.hesitation_markers import extract_hesitation_markers_vector

        return extract_hesitation_markers_vector(record.full_transcript if record else "").tolist()

    def feature_names(self):
        from src.features.hesitation_markers import HESITATION_MARKER_NAMES

        return list(HESITATION_MARKER_NAMES)


class AudioEmbeddingColumn(ColumnFeaturizer):
    """Embedding por janela do grupo ``audio_embedder`` (qualquer backend) — em lotes."""

    def __init__(self, cfg: Any, batch_windows: int = 512) -> None:
        self.cfg = cfg
        self.batch_windows = int(batch_windows)
        self._embedder = None

    @property
    def column(self) -> str:
        return f"audio_emb_{self.cfg.audio_embedder.name}"

    @property
    def embedder(self):
        if self._embedder is None:
            from src.features.builder import build_audio_embedder

            self._embedder = build_audio_embedder(self.cfg)
        return self._embedder

    def compute(self, windows, records):
        from tqdm import tqdm

        from src.features.builder import load_window_waveforms

        audio_dir = Path(self.cfg.data.paths.interim_dir) / "Audio"
        sr = int(self.cfg.data.audio.sample_rate)
        rows: list[list[float]] = []
        for a in tqdm(range(0, len(windows), self.batch_windows), desc=self.column, unit="lote"):
            batch = windows[a : a + self.batch_windows]
            emb = self.embedder.extract(load_window_waveforms(batch, audio_dir, sr))
            rows.extend(emb.tolist())
        return rows

    def feature_names(self):
        return self.embedder.feature_names()


class FaceCropsColumn(ColumnFeaturizer):
    """``[face ‖ eyes ‖ mouth]`` por janela (ver ``vision_embedder.FaceCropEmbedder``)."""

    needs_records = True  # .mp4 → fps (frame ↔ tempo)

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._crops = None

    @property
    def column(self) -> str:
        return f"face_crops_{self.cfg.vision_embedder.name}"

    @property
    def crops(self):
        if self._crops is None:
            from src.features.vision_embedder import FaceCropEmbedder, HFImageEmbedder

            v = self.cfg.vision_embedder
            backbone = HFImageEmbedder(
                model_name=v.model_name,
                pooling=v.get("pooling", "cls"),
                batch_size=v.get("batch_size", 64),
                device=self.cfg.device,
            )
            frames_root = v.get("frames_root") or (
                Path(self.cfg.data.paths.data_root) / "cropped-aligned-faces"
            )
            self._crops = FaceCropEmbedder(
                backbone,
                frames_root=frames_root,
                crops=_plain(v.get("crops")),
                sample_fps=float(v.get("sample_fps", 1.0)),
                default_fps=float(v.get("default_fps", 30.0)),
            )
        return self._crops

    def compute(self, windows, records):
        from tqdm import tqdm

        by_video: dict[str, list[int]] = {}
        for i, w in enumerate(windows):
            by_video.setdefault(w.video_id, []).append(i)
        out = np.zeros((len(windows), self.crops.dim), dtype=np.float32)
        missing = 0
        for vid, idxs in tqdm(by_video.items(), desc=self.column, unit="vídeo"):
            rec = (records or {}).get(vid)
            spans = [(windows[i].t0, windows[i].t1) for i in idxs]
            emb = self.crops.embed_video(vid, spans, rec.video_path if rec else None)
            missing += int(not emb.any())
            out[idxs] = emb
        if missing:
            log.warning(f"{missing}/{len(by_video)} vídeos sem frames de rosto (zeros).")
        return out.tolist()

    def feature_names(self):
        return self.crops.feature_names()


class SceneColumn(VideoLevelColumn):
    """Embedding de CENA do vídeo inteiro (VideoMAE congelado), replicado nas janelas."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._embedder = None

    @property
    def column(self) -> str:
        return f"scene_emb_{self.cfg.scene_embedder.name}"

    @property
    def embedder(self):
        if self._embedder is None:
            from src.features.vision_embedder import SceneEmbedder

            s = self.cfg.scene_embedder
            self._embedder = SceneEmbedder(
                model_name=s.model_name,
                num_frames=int(s.get("num_frames", 16)),
                pooling=s.get("pooling", "mean"),
                device=self.cfg.device,
            )
        return self._embedder

    def compute_video(self, record):
        if record is None:
            return [0.0] * self.embedder.dim
        return self.embedder.extract([record.video_path])[0].tolist()

    def feature_names(self):
        return self.embedder.feature_names()


def _plain(node: Any) -> Any:
    """``DictConfig``/``ListConfig`` → containers Python (``None`` passa direto)."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


# Registry {nome em `columns=[...]`: factory(cfg) -> ColumnFeaturizer}
COLUMN_FEATURIZERS: dict[str, Callable[[Any], ColumnFeaturizer]] = {
    "transcript": lambda cfg: TranscriptColumn(),
    "asr_timing": lambda cfg: AsrTimingColumn(),
    "hesitation_markers": lambda cfg: HesitationMarkersColumn(),
    "audio": AudioEmbeddingColumn,
    "face_crops": FaceCropsColumn,
    "scene": SceneColumn,
}


def build_column_featurizer(name: str, cfg: Any) -> ColumnFeaturizer:
    """Instancia o featurizer registrado em ``name`` (ver :data:`COLUMN_FEATURIZERS`)."""
    if name not in COLUMN_FEATURIZERS:
        raise KeyError(f"Coluna '{name}' desconhecida. Disponíveis: {sorted(COLUMN_FEATURIZERS)}")
    return COLUMN_FEATURIZERS[name](cfg)
