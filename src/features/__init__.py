"""Feature engineering para o pipeline BAH (áudio + texto · síntese).

Expõe:
- TextEmbedder:      embeddings de texto via HuggingFace (RoBERTa-emotion default).
- AudioEmbedder:     FACTORY por backend — librosa (prosódia) | wav2vec2 | hubert.
- TabularFeaturizer: metadados do participante + question_type + prosódia da janela.
- FeatureBuilder:    monta 1 linha/janela e escreve Parquet (README §6.2).
- build_feature_components: mapeia a Config Hydra → componentes da FASE 3.
"""

from __future__ import annotations

from src.features.audio_embedder import AudioEmbedder, create_audio_embedder
from src.features.builder import FeatureBuilder, build_feature_components
from src.features.tabular import TabularFeaturizer
from src.features.text_embedder import TextEmbedder

__all__ = [
    "TextEmbedder",
    "AudioEmbedder",
    "create_audio_embedder",
    "TabularFeaturizer",
    "FeatureBuilder",
    "build_feature_components",
]
