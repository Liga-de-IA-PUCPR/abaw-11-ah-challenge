"""FeatureBuilder: escreve o cache de features em **Parquet, 1 linha por janela**.

Produz o Parquet canônico do README §6.2 (Polars/PyArrow), com as colunas:

    id · window_idx · t0 · t1 · participant_id · question_type
    · audio_emb (list<float32>[d_audio]) · text_emb (list<float32>[d_text])
    · tabular (list<float32>[d_tab]) · label (int8) · video_label (int8) · split (str)

- **RF** (sklearn) lê o Parquet e achata ``X = [audio_emb ‖ text_emb ‖ tabular]``.
- **Cross-attention** agrupa por ``id`` → sequência ``(T, d_audio)``/``(T, d_text)``.
- O split participant-wise viaja na coluna ``split`` — ``datasets.load_split`` filtra por ela.

Fonte dos dados de janela
-------------------------
``build`` consome a lista de ``WindowSample`` do índice de janelas (FASE 2) e **carrega
internamente** o waveform de cada janela, cortado em ``[t0, t1]`` de
``audio_dir/<pid>/<stem>.flac`` (mesma convenção do ``preprocess``) via
:func:`~src.data.audio_io.load_segment`. O ``video_label`` (= ``global_ah``) e o ``split``
já vêm no próprio ``WindowSample`` (herdados do ``VideoRecord`` na indexação).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.data.audio_io import load_segment
from src.data.schema import VideoRecord, WindowSample
from src.features.audio_embedder import AudioEmbedder
from src.features.tabular import TabularFeaturizer
from src.features.text_embedder import TextEmbedder
from src.logger import get_logger

log = get_logger("features.builder")


def _jsonable(obj: object) -> object:
    """Fallback de ``json.dumps``: converte containers OmegaConf em tipos puros.

    ``feature_set``/``agg_stats`` chegam do Hydra como ``ListConfig`` (não ``list``),
    que o ``json`` não serializa. Converte ListConfig/DictConfig → list/dict.
    """
    from omegaconf import OmegaConf

    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return str(obj)


# Comprimento mínimo (s) do waveform da janela — evita erros do librosa em segmentos
# vazios/curtos no fim do áudio (ex.: delta dos MFCC exige >= 9 frames). 0,5 s @ 16 kHz
# = 8000 amostras ≈ 16 frames (folga sobre o n_fft padrão e a janela do delta).
_MIN_WAVE_S = 0.5

# Colunas EXATAS do Parquet (README §6.2 + ``split`` p/ o filtro participant-wise).
PARQUET_COLUMNS: list[str] = [
    "id",
    "window_idx",
    "t0",
    "t1",
    "participant_id",
    "question_type",
    "audio_emb",
    "text_emb",
    "tabular",
    "label",
    "video_label",
    "split",
]


class FeatureBuilder:
    """Monta o Parquet de features (1 linha por janela, README §6.2).

    Attributes:
        text_embedder / audio_embedder / tabular: extratores configurados.
        audio_dir: raiz dos ``.flac`` extraídos (``data/interim/Audio``).
        sample_rate: taxa dos ``.flac`` (= ``data.audio.sample_rate``).
        cfg_hash: hash determinístico da config dos embedders (vai no sidecar JSON).
    """

    def __init__(
        self,
        text_embedder: TextEmbedder,
        audio_embedder: AudioEmbedder,
        tabular: TabularFeaturizer,
        audio_dir: str | Path,
        sample_rate: int = 16000,
        text_featurizer=None,
    ) -> None:
        """Inicializa o FeatureBuilder.

        Args:
            text_embedder: :class:`TextEmbedder` configurado.
            audio_embedder: :class:`AudioEmbedder` (factory) configurado.
            tabular: :class:`TabularFeaturizer` (deve estar fitted no train).
            audio_dir: raiz dos ``.flac``; o waveform de cada janela é cortado de
                ``audio_dir/<pid>/<stem>.flac`` em ``[t0, t1]``.
            sample_rate: taxa dos ``.flac`` (= ``data.audio.sample_rate``).
            text_featurizer: :class:`TextFeaturizer` opcional p/ EARLY FUSION — quando
                fornecido, suas colunas são concatenadas ao ``text_emb`` (passam pela atenção).
        """
        self.text_embedder = text_embedder
        self.audio_embedder = audio_embedder
        self.tabular = tabular
        self.text_featurizer = text_featurizer
        self.audio_dir = Path(audio_dir)
        self.sample_rate = int(sample_rate)
        self.cfg_hash = self._config_hash()

    # =========================================================================
    # Build
    # =========================================================================

    def build(
        self,
        windows: list[WindowSample],
        out_path: str | Path,
        records: list[VideoRecord] | None = None,
    ) -> pl.DataFrame:
        """Constrói o Parquet único de features (todas as janelas, todos os splits).

        Carrega o waveform de cada janela (cortado em ``[t0, t1]``), extrai
        ``text_emb``/``audio_emb``/``tabular`` e escreve UMA linha por janela com as
        colunas do README §6.2 + ``split``. O ``video_label`` vem do próprio
        ``WindowSample`` (``records`` é fallback opcional por ``video_id``).

        Args:
            windows: ``WindowSample`` (FASE 2), na ordem das janelas.
            out_path: destino do Parquet (``cfg.data.paths.parquet_path``).
            records: opcional — fallback de ``video_label`` (``global_ah``) por vídeo.

        Returns:
            ``polars.DataFrame`` escrito em ``out_path``.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(windows)
        log.info(f"Construindo features para {n} janelas → {out_path.name}...")

        # --- waveforms (cortados em [t0, t1] do .flac do vídeo) ----------------
        waveforms = self._load_waveforms(windows)

        # --- embeddings + tabular ----------------------------------------------
        text_emb = self.text_embedder.extract([w.text for w in windows])  # (n, d_text)
        # Early fusion: concatena o TextFeaturizer AO text_emb (agrupado por vídeo → broadcast).
        if self.text_featurizer is not None and len(windows):
            text_feats = self.text_featurizer.extract(
                [w.text for w in windows], [w.video_id for w in windows]
            )
            text_emb = np.concatenate([text_emb, text_feats], axis=1)  # (n, d_text + d_textfeat)
        audio_emb = self.audio_embedder.extract(waveforms)  # (n, d_audio)
        tab = self.tabular.transform(windows, waveforms)  # (n, d_tab)

        # --- video_label: do WindowSample; fallback nos VideoRecord ------------
        rec_labels: dict[str, int] = {
            r.video_id: int(r.global_ah) for r in (records or []) if r.global_ah is not None
        }

        def _vlabel(w: WindowSample) -> int:
            if w.video_label is not None:
                return int(w.video_label)
            return rec_labels.get(w.video_id, -1)

        df = pl.DataFrame(
            {
                "id": [w.video_id for w in windows],
                "window_idx": np.arange(n, dtype=np.int32) if n else [],
                "t0": [float(w.t0) for w in windows],
                "t1": [float(w.t1) for w in windows],
                "participant_id": [w.participant_id for w in windows],
                "question_type": [w.question_type for w in windows],
                "audio_emb": [row.tolist() for row in audio_emb],
                "text_emb": [row.tolist() for row in text_emb],
                "tabular": [row.tolist() for row in tab],
                "label": [int(w.label) if w.label is not None else -1 for w in windows],
                "video_label": [_vlabel(w) for w in windows],
                "split": [w.split for w in windows],
            },
            schema_overrides={
                "window_idx": pl.Int32,
                "t0": pl.Float32,
                "t1": pl.Float32,
                "label": pl.Int8,
                "video_label": pl.Int8,
            },
        ).select(PARQUET_COLUMNS)

        df.write_parquet(out_path)
        self._save_sidecar(out_path)
        log.info(
            f"Parquet salvo: {out_path.name} | {n} janelas, "
            f"d_text={self._text_dim()}, d_audio={self.audio_embedder.dim}, "
            f"d_tab={len(self.tabular.feature_names())}"
        )
        return df

    # =========================================================================
    # Áudio por janela
    # =========================================================================

    def _load_waveforms(self, windows: list[WindowSample]) -> list[np.ndarray]:
        """Corta o waveform de cada janela de ``audio_dir/<pid>/<stem>.flac`` em ``[t0, t1]``.

        Mesma convenção de caminho do ``preprocess`` (que extraiu os ``.flac``). Janelas
        com áudio ausente ou curto demais recebem silêncio mínimo (evita erro do librosa).
        """
        waveforms: list[np.ndarray] = []
        min_len = int(self.sample_rate * _MIN_WAVE_S)
        missing = 0
        for w in windows:
            stem = Path(w.video_id).stem  # <file>_Video
            flac = self.audio_dir / w.participant_id / f"{stem}.flac"
            if flac.exists():
                seg = load_segment(flac, w.t0, w.t1, sr=self.sample_rate)
            else:
                seg = np.zeros(0, dtype=np.float32)
                missing += 1
            if seg.size < min_len:  # pad p/ um piso mínimo (silêncio à direita)
                padded = np.zeros(min_len, dtype=np.float32)
                padded[: seg.size] = seg
                seg = padded
            waveforms.append(seg)
        if missing:
            log.warning(f"{missing}/{len(windows)} janelas sem .flac sob {self.audio_dir}")
        return waveforms

    # =========================================================================
    # Load / sidecar / cache key
    # =========================================================================

    @staticmethod
    def load(out_path: str | Path) -> pl.DataFrame:
        """Lê o Parquet de features escrito por :meth:`build`."""
        df = pl.read_parquet(out_path)
        log.info(f"Parquet carregado: {Path(out_path).name} ({df.height} janelas)")
        return df

    def _text_names(self) -> list[str]:
        """Nomes do grupo 'text' — inclui o TextFeaturizer se em early fusion."""
        names = list(self.text_embedder.feature_names())
        if self.text_featurizer is not None:
            names += self.text_featurizer.feature_names()
        return names

    def _text_dim(self) -> int:
        d = int(self.text_embedder.dim)
        if self.text_featurizer is not None:
            d += int(self.text_featurizer.dim)
        return d

    def _all_feature_names(self) -> dict[str, list[str]]:
        return {
            "text": self._text_names(),
            "audio": self.audio_embedder.feature_names(),
            "tabular": self.tabular.feature_names(),
        }

    def _save_sidecar(self, out_path: str | Path) -> None:
        """Grava um JSON ao lado do Parquet com ``feature_names``, ``dims`` e ``cfg_hash``."""
        meta_path = Path(out_path).with_suffix(".json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "cfg_hash": self.cfg_hash,
                    "feature_names": self._all_feature_names(),
                    "dims": {
                        "d_text": self._text_dim(),
                        "d_audio": self.audio_embedder.dim,
                        "d_tab": len(self.tabular.feature_names()),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _config_hash(self) -> str:
        """Hash determinístico da config dos embedders (informativo, vai no sidecar)."""
        payload = {
            "text": {
                "model_name": self.text_embedder.model_name,
                "pooling": self.text_embedder.pooling,
                "max_length": self.text_embedder.max_length,
                "normalize": self.text_embedder.normalize,
                "dim": self.text_embedder.dim,
                "fuse_text_features": self.text_featurizer is not None,
                "text_features_names": (
                    self.text_featurizer.feature_names()
                    if self.text_featurizer is not None
                    else None
                ),
            },
            "audio": {
                "backend": getattr(self.audio_embedder, "backend", "librosa"),
                "model_name": getattr(self.audio_embedder, "model_name", None),
                "feature_set": getattr(self.audio_embedder, "feature_set", None),
                "n_mfcc": getattr(self.audio_embedder, "n_mfcc", None),
                "agg_stats": getattr(self.audio_embedder, "agg_stats", None),
                "hesitation": getattr(self.audio_embedder, "hesitation_cfg", None),
                "dim": self.audio_embedder.dim,
            },
            "tabular": {
                "silence_rms_threshold": self.tabular.silence_rms_threshold,
                "use_question_type": getattr(self.tabular, "use_question_type", True),
                "use_metadata": getattr(self.tabular, "use_metadata", True),
                "use_prosody": getattr(self.tabular, "use_prosody", True),
                "use_hesitation": getattr(self.tabular, "use_hesitation", False),
                "hesitation": getattr(self.tabular, "hesitation", None),
                "use_text_features": getattr(self.tabular, "use_text_features", False),
                "text_features": getattr(self.tabular, "text_features", None),
            },
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_jsonable)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# Factory: monta os componentes de feature a partir de uma Config Hydra
# =============================================================================


def build_feature_components(
    cfg,
    train_windows: list[WindowSample],
) -> FeatureBuilder:
    """Constrói o ``FeatureBuilder`` (+ os três extratores) a partir da Config Hydra.

    Centraliza o mapeamento ``cfg`` (README §7) → componentes da FASE 3. O
    ``TabularFeaturizer`` é **fitted nas janelas de treino** (vocabulários sem vazamento)
    antes de qualquer ``build``. Os embedders deep honram ``cfg.device`` (CPU/MPS/CUDA).

    Args:
        cfg: Config do experimento (grupos ``text_embedder``, ``audio_embedder``, ``data``).
        train_windows: ``WindowSample`` do split de treino p/ o fit do tabular.

    Returns:
        ``FeatureBuilder`` pronto para ``build(windows, out_path)``.
    """
    from src.features.audio_embedder import create_audio_embedder

    text_embedder = TextEmbedder(
        model_name=cfg.text_embedder.model_name,
        pooling=cfg.text_embedder.pooling,
        max_length=cfg.text_embedder.max_length,
        batch_size=cfg.text_embedder.batch_size,
        normalize=cfg.text_embedder.get("normalize", True),
        device=cfg.device,
        trust_remote_code=bool(cfg.text_embedder.get("trust_remote_code", False)),
    )
    audio_embedder = create_audio_embedder(
        backend=cfg.audio_embedder.backend,
        model_name=getattr(cfg.audio_embedder, "model_name", None),
        feature_set=getattr(cfg.audio_embedder, "feature_set", None),
        n_mfcc=getattr(cfg.audio_embedder, "n_mfcc", 20),
        agg_stats=getattr(cfg.audio_embedder, "agg_stats", None),
        hesitation=getattr(cfg.audio_embedder, "hesitation", None),
        sample_rate=cfg.data.audio.sample_rate,
        batch_size=getattr(cfg.audio_embedder, "batch_size", 8),
        device=cfg.device,
    )
    tabular = TabularFeaturizer.from_config(
        getattr(cfg.data, "tabular", None),
        sample_rate=cfg.data.audio.sample_rate,
        device=cfg.device,
    )
    tabular.fit(train_windows)

    # Early fusion opcional: embute o TextFeaturizer NO text_emb (passa pela atenção do cross-attn).
    text_featurizer = None
    if bool(cfg.text_embedder.get("fuse_text_features", False)):
        from src.features.text_features import TextFeaturizer

        if getattr(tabular, "use_text_features", False):
            log.warning(
                "text_embedder.fuse_text_features (early) E data.tabular.use_text_features (late) "
                "ligados juntos — as features de texto entram DUAS vezes. Escolha um."
            )
        text_featurizer = TextFeaturizer.from_config(
            cfg.text_embedder.get("text_features", None), device=cfg.device
        )
        text_featurizer.fit([w.text for w in train_windows])

    return FeatureBuilder(
        text_embedder=text_embedder,
        audio_embedder=audio_embedder,
        tabular=tabular,
        audio_dir=Path(cfg.data.paths.interim_dir) / "Audio",
        sample_rate=cfg.data.audio.sample_rate,
        text_featurizer=text_featurizer,
    )
