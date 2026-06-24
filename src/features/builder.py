"""FeatureBuilder: escreve o cache de features em **Parquet, 1 linha por janela**.

Produz o Parquet canônico do README §6.2 (Polars/PyArrow), com as colunas EXATAS:

    id · window_idx · t0 · t1 · participant_id · question_type
    · audio_emb (list<float32>[d_audio]) · text_emb (list<float32>[d_text])
    · tabular (list<float32>[d_tab]) · label (int8) · video_label (int8)

- **RF** lê o Parquet e achata ``X = [audio_emb ‖ text_emb ‖ tabular]`` por janela.
- **Cross-attention** agrupa por ``id`` → sequência ``(T, d_audio)``/``(T, d_text)``.

Fonte dos dados de janela
-------------------------
``build`` consome a lista de ``WindowSample`` do split (FASE 2) + os ``waveforms``
correspondentes (cortados em ``[t0, t1]`` por ``audio_io.load_segment``) + os
``VideoRecord`` (fonte de ``global_ah`` → ``video_label``). ``global_ah`` é campo de
``VideoRecord`` — **não** existe em ``WindowSample.meta``.

Cache versionado por HASH da config dos embedders: trocar o modelo de texto, o
``backend``/``feature_set`` de áudio gera um Parquet novo (não recomputa o antigo).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.data.schema import VideoRecord, WindowSample
from src.features.audio_embedder import AudioEmbedder
from src.features.tabular import TabularFeaturizer
from src.features.text_embedder import TextEmbedder
from src.logger import get_logger

log = get_logger("features.builder")

# Colunas EXATAS do README §6.2 (ordem canônica do Parquet).
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
]


class FeatureBuilder:
    """Monta e cacheia o Parquet de features (1 linha por janela, README §6.2).

    Attributes:
        cache_dir: Diretório de cache (``data/processed/``).
        text_embedder / audio_embedder / tabular: extratores configurados.
        cfg_hash: hash da config dos embedders (versiona o arquivo).
    """

    def __init__(
        self,
        text_embedder: TextEmbedder,
        audio_embedder: AudioEmbedder,
        tabular: TabularFeaturizer,
        cache_dir: str | Path = "data/processed",
    ) -> None:
        """Inicializa o FeatureBuilder.

        Args:
            text_embedder: :class:`TextEmbedder` configurado.
            audio_embedder: :class:`AudioEmbedder` (factory) configurado.
            tabular: :class:`TabularFeaturizer` (deve estar fitted no train).
            cache_dir: Onde salvar/carregar os Parquets.
        """
        self.text_embedder = text_embedder
        self.audio_embedder = audio_embedder
        self.tabular = tabular
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cfg_hash = self._config_hash()

    # =========================================================================
    # Build
    # =========================================================================

    def build(
        self,
        windows: list[WindowSample],
        split: str,
        waveforms: list[np.ndarray],
        records: list[VideoRecord] | None = None,
    ) -> pl.DataFrame:
        """Constrói (ou carrega do cache) o Parquet de features de um split.

        Escreve UMA linha por janela com as colunas do README §6.2: os embeddings
        ficam como ``list<float32>`` (não achatados) para servir aos dois modelos.

        Args:
            windows: ``WindowSample`` do split (FASE 2), na ordem das janelas.
            split: "train" | "val" | "test".
            waveforms: Waveforms da janela (cortados em ``[t0, t1]``), alinhados 1:1
                a ``windows`` — fonte do áudio/prosódia (``WindowSample`` não carrega
                waveform).
            records: ``VideoRecord`` do split, fonte de ``global_ah`` → ``video_label``.
                ``None`` no test (sem rótulos).

        Returns:
            ``polars.DataFrame`` (também persistido em ``data/processed/*.parquet``).
        """
        out_path = self._path(split)
        if out_path.exists():
            log.info(f"Cache Parquet encontrado para split={split}: {out_path.name}")
            return self.load(split)

        n = len(windows)
        log.info(f"Construindo features para split={split} ({n} janelas)...")

        # --- embeddings + tabular -------------------------------------------
        text_emb = self.text_embedder.extract([w.text for w in windows])  # (n, d_text)
        audio_emb = self.audio_embedder.extract(waveforms)  # (n, d_audio)
        tab = self.tabular.transform(windows, waveforms)  # (n, d_tab)

        # --- video_label por vídeo (global_ah dos VideoRecord) --------------
        # global_ah é campo de VideoRecord — NÃO existe em WindowSample.meta.
        video_labels: dict[str, int] = {
            rec.video_id: int(rec.global_ah) for rec in (records or []) if rec.global_ah is not None
        }

        # --- monta 1 linha por janela (README §6.2) -------------------------
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
                "video_label": [video_labels.get(w.video_id, -1) for w in windows],
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
        self._save_meta(split, names=self._all_feature_names())
        log.info(
            f"Parquet salvo: {out_path.name} | {n} janelas, "
            f"d_text={self.text_embedder.dim}, d_audio={self.audio_embedder.dim}, "
            f"d_tab={len(self.tabular.feature_names())}"
        )
        return df

    # =========================================================================
    # Load
    # =========================================================================

    def load(self, split: str) -> pl.DataFrame:
        """Carrega o Parquet de um split do cache (mesmo ``cfg_hash``)."""
        path = self._path(split)
        df = pl.read_parquet(path)
        log.info(f"Parquet carregado: {path.name} ({df.height} janelas)")
        return df

    # =========================================================================
    # Cache key / paths
    # =========================================================================

    def _path(self, split: str) -> Path:
        return self.cache_dir / f"text_audio_windows_{split}_{self.cfg_hash}.parquet"

    def _all_feature_names(self) -> dict[str, list[str]]:
        return {
            "text": self.text_embedder.feature_names(),
            "audio": self.audio_embedder.feature_names(),
            "tabular": self.tabular.feature_names(),
        }

    def _save_meta(self, split: str, names: dict[str, list[str]]) -> None:
        meta_path = self.cache_dir / f"text_audio_windows_{split}_{self.cfg_hash}.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_names": names,
                    "dims": {
                        "d_text": self.text_embedder.dim,
                        "d_audio": self.audio_embedder.dim,
                        "d_tab": len(self.tabular.feature_names()),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _config_hash(self) -> str:
        """Hash determinístico da config dos embedders (versiona o Parquet)."""
        payload = {
            "text": {
                "model_name": self.text_embedder.model_name,
                "pooling": self.text_embedder.pooling,
                "max_length": self.text_embedder.max_length,
                "normalize": self.text_embedder.normalize,
                "dim": self.text_embedder.dim,
            },
            "audio": {
                "backend": getattr(self.audio_embedder, "backend", "librosa"),
                "model_name": getattr(self.audio_embedder, "model_name", None),
                "feature_set": getattr(self.audio_embedder, "feature_set", None),
                "n_mfcc": getattr(self.audio_embedder, "n_mfcc", None),
                "agg_stats": getattr(self.audio_embedder, "agg_stats", None),
                "dim": self.audio_embedder.dim,
            },
            "tabular": {"silence_rms_threshold": self.tabular.silence_rms_threshold},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# Factory: monta os componentes de feature a partir de uma Config Hydra
# =============================================================================


def build_feature_components(
    cfg,
    train_windows: list[WindowSample],
) -> FeatureBuilder:
    """Constrói o ``FeatureBuilder`` (+ os três extratores) a partir da Config Hydra.

    Centraliza o mapeamento ``cfg`` (README §7) → componentes da FASE 3, para que a
    orquestração (FASE 6: ``mode=featurize``) tenha uma única forma de instanciar
    tudo de modo consistente (mesmo ``cfg_hash``). O ``TabularFeaturizer`` é **fitted
    nas janelas de treino** (vocabulários sem vazamento) antes de qualquer ``build``.

    Args:
        cfg: Config do experimento (grupos ``text_embedder``, ``audio_embedder``,
            ``data``). Ver ``src/conf/schema.py`` (dataclasses + ConfigStore).
        train_windows: ``WindowSample`` do split de treino p/ o fit do tabular.

    Returns:
        ``FeatureBuilder`` pronto para ``build(windows, split, waveforms, records)``.
    """
    from src.features.audio_embedder import create_audio_embedder

    text_embedder = TextEmbedder(
        model_name=cfg.text_embedder.model_name,
        pooling=cfg.text_embedder.pooling,
        max_length=cfg.text_embedder.max_length,
        batch_size=cfg.text_embedder.batch_size,
        normalize=cfg.text_embedder.normalize,
        device=cfg.device,
    )
    audio_embedder = create_audio_embedder(
        backend=cfg.audio_embedder.backend,
        model_name=getattr(cfg.audio_embedder, "model_name", None),
        feature_set=getattr(cfg.audio_embedder, "feature_set", None),
        n_mfcc=getattr(cfg.audio_embedder, "n_mfcc", 20),
        agg_stats=getattr(cfg.audio_embedder, "agg_stats", None),
        sample_rate=cfg.data.audio.sample_rate,
        batch_size=getattr(cfg.audio_embedder, "batch_size", 8),
        device=cfg.device,
    )
    tabular = TabularFeaturizer(sample_rate=cfg.data.audio.sample_rate)
    tabular.fit(train_windows)

    return FeatureBuilder(
        text_embedder=text_embedder,
        audio_embedder=audio_embedder,
        tabular=tabular,
        cache_dir=cfg.data.paths.processed_dir,
    )
