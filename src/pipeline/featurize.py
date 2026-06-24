"""Orquestração da featurização BAH (FASE 3).

``run_featurize(cfg, device)`` transforma o índice de janelas (FASE 2) no
**cache Parquet de features** (uma linha por janela — README §6.2):

    1. Carrega o índice de janelas de ``data/interim/windows_index.parquet``.
    2. Constrói os embedders (FASE 3) via *factories*:
         - TextEmbedder  (default ``cardiffnlp/twitter-roberta-base-emotion``; honra device)
         - AudioEmbedder (factory ``librosa`` | ``wav2vec2`` | ``hubert``; deep honra device)
       e o ``TabularFeaturizer`` (metadados, ``question_type``, prosódia) — *fitted*
       só nas janelas de TREINO (sem vazamento).
    3. ``FeatureBuilder`` extrai ``text_emb``/``audio_emb``/``tabular`` por janela e
       escreve o Parquet em ``data/processed/`` com as colunas do README §6.2:
       ``id, window_idx, t0, t1, participant_id, question_type, audio_emb, text_emb,
       tabular, label, video_label``.

Os embedders **deep** (RoBERTa, wav2vec2/HuBERT) recebem ``device`` (CPU/MPS/CUDA);
o ``librosa`` roda em CPU/numpy. Reentrante: ``cfg.data.force=True`` recomputa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from src.logger import get_logger

log = get_logger("pipeline.featurize")


def run_featurize(cfg: DictConfig, *, device: torch.device) -> dict[str, Any]:
    """Janelas → embedders → tabular → Parquet (1 linha/janela) — FASE 3.

    Args:
        cfg: Config Hydra (usa ``cfg.text_embedder``, ``cfg.audio_embedder``,
            ``cfg.data``).
        device: Device resolvido por ``resolve_device(cfg.device)`` (MPS no Mac).

    Returns:
        Resumo com ``n_windows``, ``parquet_path`` e as dimensões
        ``d_text``/``d_audio``/``d_tab``.
    """
    from src.data.windowing import load_window_index
    from src.features.audio_embedder import create_audio_embedder
    from src.features.builder import FeatureBuilder
    from src.features.tabular import TabularFeaturizer
    from src.features.text_embedder import TextEmbedder

    data = cfg.data
    force = bool(data.get("force", False))

    interim_dir = Path(data.paths.interim_dir)
    processed_dir = Path(data.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = processed_dir / "text_audio_windows.parquet"
    if parquet_path.exists() and not force:
        log.info(f"Cache encontrado: {parquet_path} (use 'data.force=true' p/ recomputar).")

    # --- 1. Índice de janelas (FASE 2) -----------------------------------------
    windows_index = interim_dir / "windows_index.parquet"
    if not windows_index.exists():
        raise FileNotFoundError(
            f"Índice de janelas ausente: {windows_index}. "
            "Rode 'python main.py mode=preprocess' antes."
        )
    windows = load_window_index(windows_index)
    log.info(f"{len(windows)} janelas carregadas de {windows_index}.")

    # --- 2. Embedders (FASE 3) — deep honram o device --------------------------
    text_embedder = TextEmbedder(
        model_name=cfg.text_embedder.model_name,      # roberta-emotion default
        pooling=cfg.text_embedder.pooling,
        device=device,
    )
    audio_embedder = create_audio_embedder(           # factory: librosa | wav2vec2 | hubert
        backend=cfg.audio_embedder.backend,
        model_name=cfg.audio_embedder.get("model_name", None),
        feature_set=cfg.audio_embedder.get("feature_set", None),
        n_mfcc=cfg.audio_embedder.get("n_mfcc", 20),
        agg_stats=cfg.audio_embedder.get("agg_stats", None),
        sample_rate=cfg.data.audio.sample_rate,
        batch_size=cfg.audio_embedder.get("batch_size", 8),
        device=device,
    )
    # TabularFeaturizer fitted SÓ no treino (vocabulários sem vazamento).
    # NB: ``split`` NÃO é campo de ``WindowSample`` (README §6.1) e NÃO está em
    # ``record.meta`` (demográficos). Por isso o ``WindowGenerator`` (FASE 2) injeta o
    # split do ``VideoRecord`` no ``meta`` da janela — ``meta={**record.meta,
    # "split": record.split}`` — de modo que o filtro abaixo de fato selecione o treino
    # (sem essa injeção, ``w.meta.get("split")`` seria sempre ``None`` → zero janelas).
    train_windows = [w for w in windows if w.meta.get("split") == "train"]
    tabular = TabularFeaturizer.from_config(cfg.data.tabular).fit(train_windows)

    log.info(
        f"Embedders: text='{text_embedder.name}'(d={text_embedder.dim}) · "
        f"audio='{audio_embedder.name}'(d={audio_embedder.dim}) · "
        f"tabular(d={tabular.dim}) · device={device}."
    )

    # --- 3. FeatureBuilder → Parquet (README §6.2) -----------------------------
    builder = FeatureBuilder(
        text_embedder=text_embedder,
        audio_embedder=audio_embedder,
        tabular=tabular,
        audio_dir=interim_dir / "Audio",
        batch_size=cfg.data.get("featurize_batch_size", 32),
    )
    builder.build(windows, out_path=parquet_path)
    log.info(f"Parquet de features escrito: {parquet_path}.")

    return {
        "n_windows": len(windows),
        "parquet_path": str(parquet_path),
        "d_text": text_embedder.dim,
        "d_audio": audio_embedder.dim,
        "d_tab": tabular.dim,
    }
