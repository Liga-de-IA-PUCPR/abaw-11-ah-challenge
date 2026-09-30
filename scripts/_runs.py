"""Runs de entrada dos scripts de roteamento/fusão (CA ⊕ GNN ⊕ face).

Os scripts do Rodrigo fixam no código os run dirs da máquina dele (7 seeds CA
``20260713_1752*``, GNN ``20260713_162753``, face ``20260713_201255``). Estas funções
deixam sobrepor por variável de ambiente, sem mudar a interface dos scripts:

    BAH_CA_RUNS=outputs/cross_attention/A,outputs/cross_attention/B   (ou só os timestamps)
    BAH_GNN_RUN=outputs/hetero_gnn_contrastive/20260713_162753
    BAH_FACE_RUN=outputs/face_gnn_ts/<timestamp>

Sem as variáveis, valem os defaults originais. Cada run precisa de
``eval_{train,val,test}/predictions.csv`` (gerados por ``mode=evaluate`` —
``make eval-members``).
"""

from __future__ import annotations

import os
from pathlib import Path


def ca_seeds(default: list[str]) -> list[str]:
    """Timestamps dos runs CA (sob ``outputs/cross_attention/``); ``BAH_CA_RUNS`` sobrepõe."""
    raw = os.environ.get("BAH_CA_RUNS", "").strip()
    if not raw:
        return list(default)
    return [Path(x.strip()).name for x in raw.replace("\n", ",").split(",") if x.strip()]


def run_dir(env: str, default: str) -> str:
    """Run dir de um membro (``BAH_GNN_RUN`` / ``BAH_FACE_RUN``) com fallback no default."""
    return os.environ.get(env, "").strip() or default
