"""Construção de grafos heterogêneos por vídeo para o modelo GNN.

Cada vídeo vira um :class:`torch_geometric.data.HeteroData` com:

- **Nós ``audio`` / ``text``** — uma janela deslizante por nó (T nós).
- **Nó ``fused``** (opcional) — fusão MultimodalBlock por janela.
- **Arestas temporais** — cadeias intra-modais + ``fused→fused``.
- **Arestas ``aligns``** — alinhamento cross-modal por janela.
- **Nó ``video``** — readout global + features tabulares.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

# Grafo legado (hetero_gnn, GAE pretrain)
BAH_NODE_TYPES = ["audio", "text", "video"]
BAH_EDGE_TYPES: list[tuple[str, str, str]] = [
    ("audio", "temporal", "audio"),
    ("text", "temporal", "text"),
    ("audio", "aligns", "text"),
    ("audio", "reports", "video"),
    ("text", "reports", "video"),
]
BAH_GRAPH_METADATA: tuple[list[str], list[tuple[str, str, str]]] = (
    BAH_NODE_TYPES,
    BAH_EDGE_TYPES,
)

# Grafo enriquecido (multimodal_hetero_full)
BAH_FULL_NODE_TYPES = ["audio", "text", "fused", "video"]
BAH_FULL_EDGE_TYPES: list[tuple[str, str, str]] = [
    ("audio", "temporal", "audio"),
    ("text", "temporal", "text"),
    ("fused", "temporal", "fused"),
    ("audio", "aligns", "text"),
    ("fused", "aligns", "audio"),
    ("fused", "aligns", "text"),
    ("audio", "reports", "video"),
    ("text", "reports", "video"),
    ("fused", "reports", "video"),
]
BAH_FULL_GRAPH_METADATA: tuple[list[str], list[tuple[str, str, str]]] = (
    BAH_FULL_NODE_TYPES,
    BAH_FULL_EDGE_TYPES,
)


def _chain_edges(num_nodes: int, device: torch.device) -> Tensor:
    """Arestas ``i → i+1`` para sequência temporal (0 arestas se T < 2)."""
    if num_nodes < 2:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    src = torch.arange(num_nodes - 1, device=device)
    dst = torch.arange(1, num_nodes, device=device)
    return torch.stack([src, dst])


def _same_index_edges(num_nodes: int, device: torch.device) -> Tensor:
    """Arestas ``i → i`` (alinhamento na mesma janela)."""
    idx = torch.arange(num_nodes, device=device)
    return torch.stack([idx, idx])


def _to_video_edges(num_nodes: int, device: torch.device) -> Tensor:
    """Arestas ``window_i → video_0``."""
    src = torch.arange(num_nodes, dtype=torch.long, device=device)
    dst = torch.zeros(num_nodes, dtype=torch.long, device=device)
    return torch.stack([src, dst])


def build_video_hetero_graph(
    audio_seq: Tensor,
    text_seq: Tensor,
    tab_seq: Tensor | None = None,
    fused_seq: Tensor | None = None,
) -> Any:
    """Monta um ``HeteroData`` a partir das sequências de janelas de UM vídeo."""
    from torch_geometric.data import HeteroData

    if audio_seq.ndim != 2 or text_seq.ndim != 2:
        msg = "audio_seq e text_seq devem ser (T, D)."
        raise ValueError(msg)
    if audio_seq.size(0) != text_seq.size(0):
        msg = "audio_seq e text_seq devem ter o mesmo T."
        raise ValueError(msg)

    t_windows = int(audio_seq.size(0))
    device = audio_seq.device
    data = HeteroData()

    data["audio"].x = audio_seq.float()
    data["text"].x = text_seq.float()

    if fused_seq is not None and fused_seq.numel() > 0:
        if fused_seq.size(0) != t_windows:
            msg = "fused_seq deve ter o mesmo T que audio/text."
            raise ValueError(msg)
        data["fused"].x = fused_seq.float()

    if tab_seq is not None and tab_seq.numel() > 0:
        data["video"].x = tab_seq.float().mean(dim=0, keepdim=True)
    else:
        data["video"].x = torch.zeros(1, 1, device=device)

    data["audio", "temporal", "audio"].edge_index = _chain_edges(t_windows, device)
    data["text", "temporal", "text"].edge_index = _chain_edges(t_windows, device)
    data["audio", "aligns", "text"].edge_index = _same_index_edges(t_windows, device)
    data["audio", "reports", "video"].edge_index = _to_video_edges(t_windows, device)
    data["text", "reports", "video"].edge_index = _to_video_edges(t_windows, device)

    if fused_seq is not None and fused_seq.numel() > 0:
        data["fused", "temporal", "fused"].edge_index = _chain_edges(t_windows, device)
        data["fused", "aligns", "audio"].edge_index = _same_index_edges(t_windows, device)
        data["fused", "aligns", "text"].edge_index = _same_index_edges(t_windows, device)
        data["fused", "reports", "video"].edge_index = _to_video_edges(t_windows, device)

    return data


def collate_graph_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate que converte cada vídeo em grafo heterogêneo e empilha com PyG ``Batch``."""
    from torch_geometric.data import Batch

    graphs = []
    labels = []
    video_ids = []

    for item in batch:
        length = int(item["length"])
        audio = item["audio_seq"][:length]
        text = item["text_seq"][:length]
        tab = item["tab_seq"][:length]
        graphs.append(build_video_hetero_graph(audio, text, tab))
        labels.append(float(item["label"]))
        video_ids.append(item["video_id"])

    batched = Batch.from_data_list(graphs)
    label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return {
        "graph": batched,
        "label": label_tensor,
        "video_id": video_ids,
    }
