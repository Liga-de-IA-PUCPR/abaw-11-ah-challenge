"""Modelos do desafio BAH — dois na registry com tag de família.

- ``random_forest``  (family="sklearn")   : classificador de janela em CPU.
- ``cross_attention`` (family="lightning") : LightningModule sobre sequência de janelas.

O import do ``random_forest`` é eager (CPU, sem deps pesadas). O ``cross_attention``
é registrado por uma **factory lazy** (ver ``registry.py``): o módulo só é importado
quando ``create_model("cross_attention", ...)`` é chamado, evitando puxar
torch/lightning no caminho sklearn.

``LitCrossAttention`` é exposto como símbolo a nível de módulo de forma **lazy**
(via ``__getattr__``): só ao acessá-lo é que ``src.models.cross_attention`` é
importado e o ``LightningModule`` (definido como closure em ``_build_lit_module``)
fica acessível como tipo nominal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.models.random_forest import RandomForestModel  # eager: só sklearn
from src.models.registry import create_model, list_models, register_model

if TYPE_CHECKING:  # só p/ type-checkers; não importa torch/lightning em runtime
    from src.models.cross_attention import CrossAttentionFusion

__all__ = [
    "register_model",
    "create_model",
    "list_models",
    "RandomForestModel",
    "CrossAttentionFusion",
    "LitCrossAttention",
]


def __getattr__(name: str) -> Any:
    """Resolve símbolos *lazy* da cross-attention (mantém torch/lightning fora do RF).

    ``LitCrossAttention`` é uma classe closure construída em
    ``cross_attention._build_lit_module``; ao acessá-la aqui construímos uma
    instância dummy só para expor seu ``type`` como símbolo de módulo.
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
