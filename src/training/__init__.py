"""Treino, agregação janela→vídeo e métricas do desafio BAH.

Importa apenas o que é leve (sklearn/numpy). O ``LightningTrainer`` é importado
*lazy* pela ``factory`` para não puxar lightning no caminho RandomForest.
"""

from __future__ import annotations

from src.training.aggregation import aggregate_to_video, calibrate_threshold, video_scores
from src.training.factory import create_trainer, load_trainer
from src.training.metrics import evaluate_video_predictions, video_macro_f1
from src.training.sklearn_trainer import SklearnTrainer
from src.training.splits import SplitTable, group_kfold_indices, load_splits

__all__ = [
    "create_trainer",
    "load_trainer",
    "SklearnTrainer",
    "aggregate_to_video",
    "calibrate_threshold",
    "video_scores",
    "video_macro_f1",
    "evaluate_video_predictions",
    "SplitTable",
    "load_splits",
    "group_kfold_indices",
]
