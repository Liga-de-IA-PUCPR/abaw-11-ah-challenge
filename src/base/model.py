"""Interface abstrata do classificador de janela — família sklearn (README §6.4).

O modelo de janela recebe a matriz de features ``X = [audio_emb ‖ text_emb ‖ tabular]``
(uma linha por janela, README §6.2) e prevê A/H por janela. O default é um RandomForest
sklearn — **100% CPU, sem importar Lightning/torchmetrics**. A predição a nível de vídeo
sai da agregação das probabilidades de janela (FASE 4, ``src/training/aggregation.py``)
com limiar calibrado na validação.

⚠️ O modelo neural ``cross_attention`` NÃO implementa esta ABC (ele é um
``LightningModule`` conduzido pelo ``LightningTrainer``). Esta interface é exclusiva da
família ``sklearn``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger

log = get_logger("base.model")


class BaseModel(ABC):
    """Classe base abstrata para os classificadores de janela sklearn do BAH.

    Define a interface canônica (README §6.4). Implementações concretas (FASE 4)
    encapsulam um estimador sklearn e devem expor ``predict_proba`` retornando a
    probabilidade da classe positiva (``1`` = há A/H).
    """

    family: str = "sklearn"

    # ==========================================================================
    # Métodos abstratos
    # ==========================================================================

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> BaseModel:
        """Treina o modelo de janela.

        Args:
            X: Features ``(n_windows, d)`` float32 (``[audio ‖ text ‖ tabular]``).
            y: Rótulos de janela ``(n_windows,)`` em ``{0, 1}``.
            sample_weight: Pesos por amostra (opcional; combate desbalanceamento).

        Returns:
            ``self`` (permite encadeamento).
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prevê a probabilidade da classe positiva (``1`` = há A/H) por janela.

        Args:
            X: Features ``(n_windows, d)``.

        Returns:
            Vetor ``(n_windows,)`` de probabilidades em ``[0, 1]`` — entrada da
            agregação janela→vídeo (README §6.5).
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persiste o modelo (joblib) — estimador + metadados.

        Args:
            path: Caminho do arquivo de checkpoint (ex.: ``model.joblib``).
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> BaseModel:
        """Factory: cria o modelo a partir do dict de hiperparâmetros.

        Args:
            config: Bloco ``cfg.model`` (de ``configs/model/{name}.yaml``) como dict.

        Returns:
            Instância do modelo (ainda não treinada).
        """
        ...

    # ==========================================================================
    # Métodos concretos (override opcional)
    # ==========================================================================

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Prevê o rótulo de cada janela aplicando um limiar à probabilidade.

        Args:
            X: Features ``(n_windows, d)``.
            threshold: Limiar de decisão (default 0.5; em produção use o calibrado).

        Returns:
            Vetor ``(n_windows,)`` em ``{0, 1}``.
        """
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    def feature_importances(self) -> np.ndarray | None:
        """Importâncias de feature, se o backend as expuser.

        Returns:
            Vetor ``(d,)`` de importâncias ou ``None`` (default).
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(family='{self.family}')"
