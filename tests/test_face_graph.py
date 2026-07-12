"""Testes do grafo facial dinâmico (sem MediaPipe)."""

import torch

from src.data.face_graph import NUM_FACE_LANDMARKS, distance_adjacency, knn_edge_index


def test_knn_edges_shape():
    coords = torch.randn(NUM_FACE_LANDMARKS, 3)
    edge_index, edge_weight = knn_edge_index(coords, top_k=8)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_weight.shape[0]
    assert edge_index.shape[1] == NUM_FACE_LANDMARKS * 8


def test_distance_adjacency_rows_sum_to_one():
    coords = torch.randn(NUM_FACE_LANDMARKS, 3)
    adj = distance_adjacency(coords, top_k=8)
    row_sum = adj.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5)
