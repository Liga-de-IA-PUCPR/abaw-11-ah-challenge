"""Interface abstrata do trainer — genérica para as duas famílias (README §6.4).

``BaseTrainer`` define o contrato comum a ``SklearnTrainer`` (caminho CPU: treina o
modelo de janela, agrega janela→vídeo, calibra o limiar — tudo com sklearn, sem torch) e
a ``LightningTrainer`` (wrapper de ``L.Trainer`` + W&B p/ o ``cross_attention``,
importado *lazy*). Ambos consomem dados a nível de vídeo e reportam métricas a nível de
vídeo (Macro-F1, AP).

As assinaturas (``fit``/``evaluate``/``predict``/``save``/``load``) são a fonte de
verdade do README §6.4 e não mudam nas fases posteriores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.logger import get_logger

log = get_logger("base.trainer")


class BaseTrainer(ABC):
    """Classe base abstrata para os trainers do BAH.

    Define a interface (``fit``/``evaluate``/``predict``/``save``/``load``). Os
    ``*_data`` chegam no formato esperado por cada família:

    - **sklearn:** ``WindowMatrixView`` (matriz achatada por janela + ``video_ids`` /
      ``participant_ids`` / ``video_labels``) — README §6.2.
    - **lightning:** ``VideoSequenceDataset`` (sequência ``(T, D)`` por vídeo,
      ``key_padding_mask`` p/ ``T`` variável).

    Attributes:
        model: Modelo a treinar (``BaseModel`` sklearn ou ``LightningModule``).
        cfg: Configuração resolvida do experimento (``DictConfig`` ou dict).
        threshold_: Limiar de decisão calibrado na validação (README §6.5).
    """

    family: str = "base"

    def __init__(self, model: Any, cfg: Any):
        """Inicializa o trainer.

        Args:
            model: Modelo (instância de ``BaseModel`` ou ``LightningModule``).
            cfg: Configuração completa do experimento (``cfg`` do Hydra).
        """
        self.model = model
        self.cfg = cfg

        self.threshold_: float | None = None

    # ==========================================================================
    # Métodos abstratos (README §6.4)
    # ==========================================================================

    @abstractmethod
    def fit(self, train_data: Any, val_data: Any) -> dict[str, Any]:
        """Treina o modelo e calibra o limiar de decisão na validação.

        Args:
            train_data: Dados de treino (``WindowMatrixView`` | ``VideoSequenceDataset``).
            val_data: Dados de validação (mesmo tipo).

        Returns:
            Histórico/resumo do treino.
        """
        ...

    @abstractmethod
    def evaluate(self, data: Any) -> dict[str, float]:
        """Avalia a nível de vídeo num split.

        Args:
            data: Dados do split a avaliar.

        Returns:
            Métricas a nível de vídeo (ex.: ``{"macro_f1": ..., "average_precision": ...}``).
        """
        ...

    @abstractmethod
    def predict(self, data: Any) -> dict[str, int]:
        """Prediz o rótulo binário a nível de vídeo.

        Args:
            data: Dados a predizer (split de test, sem rótulo).

        Returns:
            Mapa ``{video_id: pred}`` com ``pred`` em ``{0, 1}`` (entrada da submissão).
        """
        ...

    @abstractmethod
    def save(self, out_dir: str | Path) -> None:
        """Persiste modelo + limiar + metadados em ``out_dir``.

        Args:
            out_dir: Diretório de saída (joblib p/ RF; ckpt Lightning p/ neural).
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, out_dir: str | Path) -> BaseTrainer:
        """Reconstrói o trainer (modelo + limiar) de um diretório salvo.

        Args:
            out_dir: Diretório produzido por :meth:`save`.

        Returns:
            Instância de trainer pronta p/ ``evaluate`` / ``predict``.
        """
        ...
