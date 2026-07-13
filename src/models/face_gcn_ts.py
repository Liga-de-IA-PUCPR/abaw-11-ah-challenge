"""Encoder GCN temporal sobre Face Mesh — grafo espacial variando no tempo (GNN4TS).

Modos temporais:
  - ``chain``: GCN espacial por janela → pool → GCN em cadeia ``t→t+1`` (baseline).
  - ``gnn4ts``: GCN espacial por nó → GCN temporal por landmark (arestas ``i@t ↔ i@t+1``),
    análogo a classificação de movimento em esqueletos (ST-GCN / GNN4TS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.data.face_graph import NUM_FACE_LANDMARKS, distance_adjacency
from src.features.face_mesh import LANDMARK_DIM

if TYPE_CHECKING:
    import torch

TemporalMode = Literal["chain", "gnn4ts"]


def _build_face_gcn_ts(
    *,
    in_channels: int = LANDMARK_DIM,
    spatial_hidden: int = 64,
    spatial_out: int = 128,
    temporal_hidden: int = 64,
    temporal_out: int = 128,
    top_k: int = 8,
    dropout: float = 0.1,
    temporal_mode: TemporalMode = "chain",
    use_velocity: bool = False,
    landmark_stride: int = 1,
    max_windows: int | None = None,
):
    import torch
    import torch.nn.functional as F
    from torch import nn

    effective_in = in_channels + (LANDMARK_DIM if use_velocity else 0)
    stride = max(1, int(landmark_stride))

    def _cap_length(length: int) -> int:
        if max_windows is None or max_windows <= 0:
            return length
        return min(length, int(max_windows))

    def _subsample(coords: torch.Tensor) -> torch.Tensor:
        if stride <= 1:
            return coords
        return coords[::stride]

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
        """GCN espacial sobre landmarks de UMA janela."""

        def __init__(self, *, pool_nodes: bool) -> None:
            super().__init__()
            self.pool_nodes = pool_nodes
            self.spatial_out = spatial_out
            self.gcn1 = DistanceGCNLayer(effective_in, spatial_hidden, top_k)
            self.gcn2 = DistanceGCNLayer(spatial_hidden, spatial_out, top_k)
            self.dropout = nn.Dropout(dropout)

        def forward(self, coords: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
            """``coords (N, 3)``, ``feat (N, C)`` → ``(spatial_out,)`` ou ``(N, spatial_out)``."""
            x = feat
            h = self.dropout(self.gcn1(x, coords))
            h = self.dropout(self.gcn2(h, coords))
            return h.mean(dim=0) if self.pool_nodes else h

    class FaceTemporalChainGCN(nn.Module):
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

    class FaceLandmarkTemporalGCN(nn.Module):
        """GCN temporal por landmark: ``(N, T, H)`` com arestas ``t↔t+1`` em cada nó."""

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

        def forward(self, node_seq: torch.Tensor) -> torch.Tensor:
            """``node_seq (N, T, H)`` → ``(N, temporal_out)``."""
            n, t, _ = node_seq.shape
            if t == 0 or n == 0:
                return node_seq.new_zeros(n, temporal_out)
            adj = self._chain_adjacency(t, node_seq.device)
            h = F.relu(self.gcn1(node_seq))
            h = self.dropout(h)
            h = torch.matmul(adj, h)
            h = F.relu(self.gcn2(h))
            h = torch.matmul(adj, h)
            return h.mean(dim=1)

    class FaceGraphTSEncoder(nn.Module):
        """Face Mesh → GCN espacial + GCN temporal (chain ou GNN4TS por landmark)."""

        def __init__(self) -> None:
            super().__init__()
            self.temporal_mode = temporal_mode
            self.use_velocity = use_velocity
            pool_nodes = temporal_mode == "chain"
            self.spatial = FaceSpatialGCN(pool_nodes=pool_nodes)
            if temporal_mode == "chain":
                self.temporal = FaceTemporalChainGCN()
            else:
                self.temporal = FaceLandmarkTemporalGCN()
            self.out_dim = temporal_out

        def _window_features(
            self,
            face_seq_b: torch.Tensor,
            t_idx: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Retorna ``coords (N, 3)`` e ``feat (N, C)`` para a janela ``t_idx``."""
            coords = _subsample(face_seq_b[t_idx])
            if not self.use_velocity:
                return coords, coords
            if t_idx == 0:
                vel = coords.new_zeros(coords.shape)
            else:
                vel = _subsample(face_seq_b[t_idx]) - _subsample(face_seq_b[t_idx - 1])
            feat = torch.cat([coords, vel], dim=-1)
            return coords, feat

        def _encode_chain(
            self,
            face_seq: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            batch_size = int(face_seq.size(0))
            outputs: list[torch.Tensor] = []
            for b in range(batch_size):
                t = _cap_length(int(lengths[b].item()))
                if t <= 0:
                    outputs.append(face_seq.new_zeros(self.out_dim))
                    continue
                window_embs: list[torch.Tensor] = []
                for w in range(t):
                    coords, feat = self._window_features(face_seq[b], w)
                    if coords.abs().sum() < 1e-8:
                        window_embs.append(face_seq.new_zeros(self.spatial.spatial_out))
                    else:
                        window_embs.append(self.spatial(coords, feat))
                seq = torch.stack(window_embs, dim=0)
                outputs.append(self.temporal(seq))
            return torch.stack(outputs, dim=0)

        def _encode_gnn4ts(
            self,
            face_seq: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            batch_size = int(face_seq.size(0))
            n_landmarks = max(1, (NUM_FACE_LANDMARKS + stride - 1) // stride)
            outputs: list[torch.Tensor] = []
            for b in range(batch_size):
                t = _cap_length(int(lengths[b].item()))
                if t <= 0:
                    outputs.append(face_seq.new_zeros(self.out_dim))
                    continue
                node_seq: list[torch.Tensor] = []
                for w in range(t):
                    coords, feat = self._window_features(face_seq[b], w)
                    if coords.abs().sum() < 1e-8:
                        node_seq.append(face_seq.new_zeros(n_landmarks, self.spatial.spatial_out))
                    else:
                        node_seq.append(self.spatial(coords, feat))
                stacked = torch.stack(node_seq, dim=1)
                landmark_embs = self.temporal(stacked)
                outputs.append(landmark_embs.mean(dim=0))
            return torch.stack(outputs, dim=0)

        def forward(
            self,
            face_seq: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            """``face_seq (B, T, N, 3)`` → ``(B, temporal_out)``."""
            if self.temporal_mode == "gnn4ts":
                return self._encode_gnn4ts(face_seq, lengths)
            return self._encode_chain(face_seq, lengths)

    return FaceGraphTSEncoder()
