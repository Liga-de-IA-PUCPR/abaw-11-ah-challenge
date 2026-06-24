"""Registry de modelos (factory pattern com tag de família).

Dois mecanismos de registro:
  - ``@register_model(name, family)``  : registro EAGER de uma classe ``BaseModel``
    (usado pelo ``random_forest``, que é leve e roda em CPU).
  - ``register_lazy(name, family, loader)`` : registro LAZY — guarda apenas uma
    função ``loader()`` que importa+devolve a classe na primeira chamada (usado
    pela ``cross_attention``, para não importar torch/lightning no caminho sklearn).

``create_model(name, cfg)`` devolve **(objeto, family)**. A ``family`` ("sklearn"|
"lightning") é consumida por ``src/training/factory.py`` para escolher o trainer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.logger import get_logger

log = get_logger("models.registry")

Family = str  # "sklearn" | "lightning"


@dataclass
class _Entry:
    """Entrada do registry: família + carregador da classe (lazy ou eager)."""

    family: Family
    loader: Callable[[], type]  # devolve a classe do modelo (importa se necessário)


# Registry global {nome: _Entry}
MODEL_REGISTRY: dict[str, _Entry] = {}


def register_model(name: str, family: Family) -> Callable[[type], type]:
    """Decorator de registro EAGER de uma classe de modelo.

    Args:
        name: nome único (deve casar com ``configs/model/<name>.yaml``)
        family: "sklearn" | "lightning"

    Example:
        >>> @register_model("random_forest", family="sklearn")
        ... class RandomForestModel(BaseModel): ...
    """

    def decorator(cls: type) -> type:
        if name in MODEL_REGISTRY:
            log.warning(f"Sobrescrevendo modelo já registrado: '{name}'")
        MODEL_REGISTRY[name] = _Entry(family=family, loader=lambda: cls)
        log.debug(f"Modelo registrado (eager): {name} [{family}] -> {cls.__name__}")
        return cls

    return decorator


def register_lazy(name: str, family: Family, loader: Callable[[], type]) -> None:
    """Registro LAZY: ``loader`` só é chamado (e o módulo importado) na criação.

    Usado pela cross-attention para evitar importar torch/lightning enquanto o
    usuário roda apenas o RandomForest.

    Args:
        name: nome único do modelo
        family: "lightning" (tipicamente)
        loader: função sem args que importa e devolve a classe do modelo
    """
    if name in MODEL_REGISTRY:
        log.warning(f"Sobrescrevendo modelo já registrado: '{name}'")
    MODEL_REGISTRY[name] = _Entry(family=family, loader=loader)
    log.debug(f"Modelo registrado (lazy): {name} [{family}]")


def create_model(name: str, config: Any) -> tuple[Any, Family]:
    """Factory: cria o modelo a partir do registry.

    Args:
        name: nome registrado (``cfg.model.name``)
        config: dict de hiperparâmetros (bloco ``cfg.model``)

    Returns:
        Tupla ``(objeto, family)``. Para ``sklearn`` o objeto é um ``BaseModel``
        já instanciado via ``from_config``; para ``lightning`` é a fachada
        ``CrossAttentionFusion`` (o ``LightningModule`` é montado pelo
        ``LightningTrainer``).

    Raises:
        KeyError: se o modelo não estiver registrado.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Modelo '{name}' não está no registry. Disponíveis: {list_models()}"
        )
    entry = MODEL_REGISTRY[name]
    cls = entry.loader()  # importa o módulo lazy aqui, se for o caso
    log.info(f"Criando modelo: {name} [{entry.family}] ({cls.__name__})")
    obj = cls.from_config(config=config)
    return obj, entry.family


def list_models() -> list[str]:
    """Lista os nomes de todos os modelos registrados."""
    return list(MODEL_REGISTRY.keys())


def get_family(name: str) -> Family:
    """Família de um modelo registrado (sem instanciar nem importar a classe)."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Modelo '{name}' não está no registry")
    return MODEL_REGISTRY[name].family


# ---------------------------------------------------------------------------
# Registro LAZY da cross-attention (NÃO importa torch/lightning aqui)
# ---------------------------------------------------------------------------
def _load_cross_attention() -> type:
    """Importa ``CrossAttentionFusion`` somente quando a cross-attention é criada."""
    from src.models.cross_attention import CrossAttentionFusion

    return CrossAttentionFusion


register_lazy("cross_attention", family="lightning", loader=_load_cross_attention)
