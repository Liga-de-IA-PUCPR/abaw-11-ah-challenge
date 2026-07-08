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

from omegaconf import DictConfig

from src.logger import get_logger

log = get_logger("pipeline.featurize")


def run_featurize(cfg: DictConfig, *, device: Any) -> dict[str, Any]:
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
    from src.features.builder import build_feature_components

    data = cfg.data
    force = bool(data.get("force", False))

    interim_dir = Path(data.paths.interim_dir)
    # Mesma chave que o datasets.load_split lê (FASE 4) — fonte única do caminho.
    parquet_path = Path(data.paths.parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Índice de janelas (FASE 2) -----------------------------------------
    # Usa data.paths.window_index (parametrizado por window.size_s) — antes hardcoded
    # como "windows_index.parquet" fixo, o que fazia small/medium/large lerem/
    # sobrescreverem o MESMO arquivo (bug corrigido junto com preprocess.py).
    windows_index = Path(data.paths.window_index)
    if not windows_index.exists():
        raise FileNotFoundError(
            f"Índice de janelas ausente: {windows_index}. Rode "
            f"'python main.py mode=preprocess data.window.size_s={data.window.size_s}' antes "
            "(ou o override de window= equivalente)."
        )
    windows = load_window_index(windows_index)
    log.info(f"{len(windows)} janelas carregadas de {windows_index}.")

    # Cache: pula a featurização cara (carrega RoBERTa/wav2vec2) se o Parquet já existe.
    if parquet_path.exists() and not force:
        log.info(f"Cache encontrado: {parquet_path} (use 'data.force=true' p/ recomputar).")
        return {"n_windows": len(windows), "parquet_path": str(parquet_path), "cached": True}

    # --- 2. Componentes da FASE 3 (embedders deep honram device; tabular fitted no treino) -
    # ``split`` é campo de WindowSample (herdado do VideoRecord via índice); o tabular é
    # fitted SÓ nas janelas de treino (vocabulários sem vazamento entre splits).
    train_windows = [w for w in windows if w.split == "train"]
    builder = build_feature_components(cfg, train_windows)
    log.info(
        f"Embedders: text d={builder.text_embedder.dim} · "
        f"audio='{builder.audio_embedder.name}' d={builder.audio_embedder.dim} · "
        f"tabular d={builder.tabular.dim} · device={device}."
    )

    # --- 3. FeatureBuilder → Parquet único (README §6.2; carrega os waveforms) --
    builder.build(windows, out_path=parquet_path)
    log.info(f"Parquet de features escrito: {parquet_path}.")

    return {
        "n_windows": len(windows),
        "parquet_path": str(parquet_path),
        "d_text": builder.text_embedder.dim,
        "d_audio": builder.audio_embedder.dim,
        "d_tab": builder.tabular.dim,
    }
