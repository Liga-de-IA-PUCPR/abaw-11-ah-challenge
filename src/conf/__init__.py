"""Configuração tipada (Hydra structured configs) do desafio BAH.

Reexporta os schemas, o registro no ConfigStore e os helpers de device para que as
demais fases importem de um único ponto: ``from src.conf import resolve_device``.
"""

from src.conf.schema import (
    AggregationConfig,
    AudioConfig,
    AudioEmbedderConfig,
    DataConfig,
    ModelConfig,
    PathsConfig,
    RootConfig,
    TabularConfig,
    TextEmbedderConfig,
    TrainerConfig,
    WandbConfig,
    WindowConfig,
    device_to_accelerator,
    register_configs,
    resolve_device,
    seed_everything,
    to_container,
)

__all__ = [
    "RootConfig",
    "DataConfig",
    "PathsConfig",
    "AudioConfig",
    "TabularConfig",
    "WindowConfig",
    "TextEmbedderConfig",
    "AudioEmbedderConfig",
    "ModelConfig",
    "TrainerConfig",
    "AggregationConfig",
    "WandbConfig",
    "resolve_device",
    "device_to_accelerator",
    "seed_everything",
    "register_configs",
    "to_container",
]
