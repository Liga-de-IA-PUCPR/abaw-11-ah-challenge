"""Grafo facial dinâmico — landmarks MediaPipe, arestas por distância (estilo GNN4TS).

Em cada instante temporal (janela), a adjacência é derivada das distâncias
euclidianas entre keypoints: vizinhos k-NN com peso ``exp(-d²/σ²)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.features.face_mesh import LANDMARK_DIM, NUM_FACE_LANDMARKS

__all__ = [
    "NUM_FACE_LANDMARKS",
    "LANDMARK_DIM",
    "distance_adjacency",
    "knn_edge_index",
]


def _pairwise_distance(coords: Tensor) -> Tensor:
    """Distâncias euclidianas ``(N, N)`` a partir de ``coords (N, D)``."""
    return torch.cdist(coords, coords, p=2)


def knn_edge_index(
    coords: Tensor,
    top_k: int,
) -> tuple[Tensor, Tensor]:
    """Índices de arestas k-NN + pesos ``exp(-d²/σ²)`` por vizinho.

    Args:
        coords: ``(N, D)`` posições dos landmarks.
        top_k: vizinhos por nó (exclui self-loop).

    Returns:
        ``edge_index`` ``(2, E)``, ``edge_weight`` ``(E,)``.
    """
    n = int(coords.size(0))
    if n < 2:
        empty = coords.new_zeros(2, 0, dtype=torch.long)
        return empty, coords.new_zeros(0)

    dist = _pairwise_distance(coords)
    dist.fill_diagonal_(float("inf"))
    k = min(int(top_k), n - 1)
    if k < 1:
        empty = coords.new_zeros(2, 0, dtype=torch.long)
        return empty, coords.new_zeros(0)

    knn_dist, knn_idx = dist.topk(k, dim=-1, largest=False)
    sigma = knn_dist.mean().clamp(min=1e-4)
    weights = torch.exp(-(knn_dist**2) / (sigma**2))

    src = torch.arange(n, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = weights.reshape(-1)
    return edge_index, edge_weight


def distance_adjacency(
    coords: Tensor,
    top_k: int,
    *,
    symmetric: bool = True,
) -> Tensor:
    """Matriz de adjacência esparsa/densa ``(N, N)`` com pesos por distância."""
    edge_index, edge_weight = knn_edge_index(coords, top_k)
    n = int(coords.size(0))
    adj = coords.new_zeros(n, n)
    if edge_index.numel() == 0:
        return adj
    adj[edge_index[0], edge_index[1]] = edge_weight
    if symmetric:
        adj = 0.5 * (adj + adj.T)
    row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return adj / row_sum
