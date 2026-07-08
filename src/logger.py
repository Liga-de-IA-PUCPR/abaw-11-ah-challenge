"""Logging estruturado para os experimentos do desafio BAH.

Fornece loggers escopados por módulo (:func:`get_logger`) — a API usada por TODO o
código (``from src.logger import get_logger``). A configuração de handlers (console +
arquivo por run) é feita pelo Hydra (``hydra/job_logging``); o W&B (FASE 5) cuida das
métricas/curvas, este logger cuida do texto.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger escopado para um módulo.

    Args:
        name: Nome do logger (tipicamente o nome do módulo, ex.: ``"features.text"``).

    Returns:
        Instância de logger.
    """
    return logging.getLogger(name)
