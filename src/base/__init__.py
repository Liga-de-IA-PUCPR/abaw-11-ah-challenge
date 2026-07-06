"""Classes abstratas (ABCs) do pipeline BAH.

Estas interfaces são a fonte de verdade do README §6 — as fases seguintes (features,
modelos, treino) as implementam sem alterar suas assinaturas.

Exporta:
- BaseEmbedder: extratores de features (texto/áudio) — §6.3
- BaseModel:    classificador de janela da família sklearn — §6.4
- BaseTrainer:  laço de treino + agregação (SklearnTrainer / LightningTrainer) — §6.4

O ``cross_attention`` (família lightning) NÃO herda de ``BaseModel`` (que é específico de
sklearn): ele é um ``LightningModule`` conduzido pelo ``LightningTrainer`` (que herda de
``BaseTrainer``). Assim Lightning fica importado *lazy* só quando ``model=cross_attention``.
"""

from src.base.embedder import BaseEmbedder
from src.base.model import BaseModel
from src.base.trainer import BaseTrainer

__all__ = ["BaseEmbedder", "BaseModel", "BaseTrainer"]
