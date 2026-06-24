"""Interface abstrata de extratores de features (embedders) — README §6.3.

Um embedder mapeia uma lista de entradas (textos ou waveforms) para uma matriz densa
``(n, dim)`` de ``float32``, expondo nomes de feature alinhados às colunas.

Subclasses concretas (FASE 3):
- ``TextEmbedder``  — encoder HuggingFace (``roberta_emotion`` default; honra ``device``).
- ``AudioEmbedder`` — factory ``librosa`` (prosódia, CPU) | ``wav2vec2`` | ``hubert`` (deep).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.logger import get_logger

log = get_logger("base.embedder")


class BaseEmbedder(ABC):
    """Classe base abstrata para todos os extratores de features.

    Define a interface canônica que as fases de feature engineering implementam. O
    contrato (README §6.3) é deliberadamente mínimo: dimensão fixa, extração em lote e
    nomes de feature por coluna.

    Attributes:
        name: Identificador curto do embedder (ex.: ``"text_roberta_emotion"``,
            ``"audio_librosa"``). Usado em nomes de feature e no hash de cache.
    """

    name: str = "base_embedder"

    # ==========================================================================
    # Métodos abstratos
    # ==========================================================================

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionalidade do vetor de saída (número de colunas de ``extract``)."""
        ...

    @abstractmethod
    def extract(self, inputs: list) -> np.ndarray:
        """Extrai features para um lote de entradas.

        Args:
            inputs: Lista de ``n`` entradas. Para ``TextEmbedder`` é ``list[str]``;
                para ``AudioEmbedder`` é ``list[np.ndarray]`` (waveforms mono 16 kHz).

        Returns:
            Matriz ``np.ndarray`` de shape ``(n, dim)`` e dtype ``float32``.
        """
        ...

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Nomes das colunas produzidas por ``extract``.

        Returns:
            Lista de comprimento ``dim`` — alinhada às colunas de ``extract``.
        """
        ...

    # ==========================================================================
    # Métodos concretos (utilidades comuns às subclasses)
    # ==========================================================================

    def _validate_output(self, feats: np.ndarray, n_inputs: int) -> np.ndarray:
        """Valida e normaliza a saída de ``extract`` para o contrato ``(n, dim)``.

        Garante dtype ``float32``, shape correto e ausência de NaN/Inf (substituídos
        por 0.0). Útil para chamar ao final de cada ``extract`` concreto.

        Args:
            feats: Matriz produzida pela subclasse.
            n_inputs: Número de entradas processadas.

        Returns:
            Matriz validada ``(n_inputs, dim)`` float32.

        Raises:
            ValueError: Se o shape não corresponder a ``(n_inputs, dim)``.
        """
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape != (n_inputs, self.dim):
            raise ValueError(
                f"{self.name}: shape de saída {feats.shape} != esperado "
                f"({n_inputs}, {self.dim})"
            )
        if not np.all(np.isfinite(feats)):
            n_bad = int((~np.isfinite(feats)).sum())
            log.warning(f"{self.name}: {n_bad} valores não-finitos zerados.")
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

    def __len__(self) -> int:
        """Conveniência: ``len(embedder) == embedder.dim``."""
        return self.dim

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.dim})"
