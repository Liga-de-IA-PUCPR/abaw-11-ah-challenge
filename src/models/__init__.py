"""Modelos do desafio BAH — registry com tag de família.

Família ``sklearn`` (CPU, sem Lightning):
- ``random_forest``       : RandomForestClassifier — baseline robusto.
- ``xgboost``             : XGBClassifier — gradient boosting, forte em competições.
- ``lightgbm``            : LGBMClassifier — gradient boosting rápido.
- ``extra_trees``         : ExtraTreesClassifier — splits aleatórios, menos overfitting.
- ``logistic_regression`` : LogReg com StandardScaler — baseline linear forte.
- ``catboost``            : CatBoostClassifier — boosting com árvores oblivious (lazy import).
- ``mlp``                 : MLPClassifier com StandardScaler — rede densa simples.
- ``stacking``            : StackingClassifier RF+XGB+LGBM → LogReg meta.
- ``stacking_catboost``   : RF+XGB+LGBM+CatBoost → LogReg meta.
- ``stacking_cat_rf``     : CatBoost+RF → LogReg meta (sem XGB/LGBM — versão reduzida).

Família ``lightning`` (GPU/MPS):
- ``cross_attention`` : LightningModule com cross-attention temporal (sequência de janelas).

Modelos sklearn são importados **eager** (sem deps pesadas, exceto catboost que é lazy).
O ``cross_attention`` é registrado via factory **lazy** para evitar importar torch/lightning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.models.catboost_model import (
    CatBoostModel,  # eager: import class; catboost lazy no from_config
)
from src.models.extra_trees import ExtraTreesModel  # eager: só sklearn
from src.models.lightgbm_model import LightGBMModel  # eager: só lightgbm
from src.models.logistic_regression import LogisticRegressionModel  # eager: só sklearn
from src.models.mlp_model import MLPModel  # eager: só sklearn
from src.models.random_forest import RandomForestModel  # eager: só sklearn
from src.models.registry import create_model, list_models, register_model
from src.models.stacking_model import StackingModel  # eager: deps lazy no from_config
from src.models.xgboost_model import XGBoostModel  # eager: só xgboost

if TYPE_CHECKING:  # só p/ type-checkers; não importa torch/lightning em runtime
    from src.models.cross_attention import CrossAttentionFusion

__all__ = [
    "register_model",
    "create_model",
    "list_models",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "ExtraTreesModel",
    "LogisticRegressionModel",
    "CatBoostModel",
    "MLPModel",
    "StackingModel",
    "CrossAttentionFusion",
    "LitCrossAttention",
]


def __getattr__(name: str) -> Any:
    """Resolve símbolos *lazy* da cross-attention (mantém torch/lightning fora do RF).

    LitCrossAttention é uma classe closure construída em
    cross_attention._build_lit_module; ao acessá-la aqui construímos uma
    instância dummy só para expor seu type como símbolo de módulo.
    """
    if name == "CrossAttentionFusion":
        from src.models.cross_attention import CrossAttentionFusion

        return CrossAttentionFusion
    if name == "LitCrossAttention":
        # Constrói uma instância mínima e devolve sua classe (closure interna).
        from src.models.cross_attention import _build_fusion_module, _build_lit_module

        fusion = _build_fusion_module(dim_a=1, dim_b=1, common_dim=2, num_heads=1, dropout=0.0)
        return type(_build_lit_module(fusion=fusion, lr=1e-3, weight_decay=1e-2))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
