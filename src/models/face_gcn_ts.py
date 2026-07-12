"""Encoder GCN temporal sobre Face Mesh — grafo espacial variando no tempo (GNN4TS).

Por janela temporal:
  1. Grafo espacial com 468 nós (landmarks MediaPipe).
  2. Arestas k-NN ponderadas por distância euclidiana entre pontos.
  3. GCN espacial → pooling → embedding de janela.

Sobre a sequência de janelas:
  4. GCN temporal com cadeia ``t → t+1`` (grafo dinâmico no tempo).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.data.face_graph import NUM_FACE_LANDMARKS, distance_adjacency
from src.features.face_mesh import LANDMARK_DIM

if TYPE_CHECKING:
    import torch


def _build_face_gcn_ts(
    *,
    in_channels: int = LANDMARK_DIM,
    spatial_hidden: int = 64,
    spatial_out: int = 128,
    temporal_hidden: int = 64,
    temporal_out: int = 128,
    top_k: int = 8,
    dropout: float = 0.1,
):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class DistanceGCNLayer(nn.Module):
        """Uma camada de message passing com adjacência fixada por distância."""

        def __init__(self, in_dim: int, out_dim: int, top_k: int) -> None:
            super().__init__()
            self.top_k = top_k
            self.lin = nn.Linear(in_dim, out_dim)

        def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
            adj = distance_adjacency(coords, self.top_k)
            h = adj @ x
            return F.relu(self.lin(h))

    class FaceSpatialGCN(nn.Module):
        """GCN espacial sobre 468 landmarks de UMA janela."""

        def __init__(self) -> None:
            super().__init__()
            self.spatial_out = spatial_out
            self.gcn1 = DistanceGCNLayer(in_channels, spatial_hidden, top_k)
            self.gcn2 = DistanceGCNLayer(spatial_hidden, spatial_out, top_k)
            self.dropout = nn.Dropout(dropout)

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            """``coords (468, 3)`` → embedding pooled ``(spatial_out,)``."""
            x = coords
            h = self.dropout(self.gcn1(x, coords))
            h = self.dropout(self.gcn2(h, coords))
            return h.mean(dim=0)

    class FaceTemporalGCN(nn.Module):
        """GCN temporal sobre embeddings de janela (cadeia ``t→t+1``)."""

        def __init__(self) -> None:
            super().__init__()
            self.gcn1 = nn.Linear(spatial_out, temporal_hidden)
            self.gcn2 = nn.Linear(temporal_hidden, temporal_out)
            self.dropout = nn.Dropout(dropout)

        @staticmethod
        def _chain_adjacency(length: int, device: torch.device) -> torch.Tensor:
            if length < 2:
                return torch.eye(length, device=device)
            adj = torch.zeros(length, length, device=device)
            for i in range(length - 1):
                adj[i, i + 1] = 1.0
                adj[i + 1, i] = 1.0
            adj += torch.eye(length, device=device)
            return adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        def forward(self, seq: torch.Tensor) -> torch.Tensor:
            """``seq (T, spatial_out)`` → ``(temporal_out,)``."""
            t = int(seq.size(0))
            if t == 0:
                return seq.new_zeros(temporal_out)
            adj = self._chain_adjacency(t, seq.device)
            h = F.relu(self.gcn1(seq))
            h = self.dropout(h)
            h = adj @ h
            h = F.relu(self.gcn2(h))
            h = adj @ h
            return h.mean(dim=0)

    class FaceGraphTSEncoder(nn.Module):
        """Face Mesh 468 pts → GCN espacial + GCN temporal (GNN4TS-style)."""

        def __init__(self) -> None:
            super().__init__()
            self.spatial = FaceSpatialGCN()
            self.temporal = FaceTemporalGCN()
            self.out_dim = temporal_out

        def forward(
            self,
            face_seq: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            """``face_seq (B, T, 468, 3)`` → ``(B, temporal_out)``."""
            batch_size = int(face_seq.size(0))
            outputs: list[torch.Tensor] = []
            for b in range(batch_size):
                t = int(lengths[b].item())
                if t <= 0:
                    outputs.append(face_seq.new_zeros(self.out_dim))
                    continue
                window_embs: list[torch.Tensor] = []
                for w in range(t):
                    coords = face_seq[b, w]
                    if coords.abs().sum() < 1e-8:
                        window_embs.append(face_seq.new_zeros(self.spatial.spatial_out))
                    else:
                        window_embs.append(self.spatial(coords))
                seq = torch.stack(window_embs, dim=0)
                outputs.append(self.temporal(seq))
            return torch.stack(outputs, dim=0)

    return FaceGraphTSEncoder()
