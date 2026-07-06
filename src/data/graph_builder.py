"""Construção de grafos heterogêneos por vídeo para o modelo GNN.

Cada vídeo vira um :class:`torch_geometric.data.HeteroData` com:

- **Nós ``audio`` / ``text``** — uma janela deslizante por nó (T nós).
- **Arestas temporais** — ``audio→audio`` e ``text→text`` (vizinhança local /
  sequência; análogo ao *local self-attention* da branch ``matheus-local-attn``).
- **Arestas ``aligns``** — ``audio_i → text_i`` (alinhamento por janela; análogo
  ao *cross-attention* da branch ``luiz``).
- **Nó ``video``** — nó de leitura global (1 por vídeo) conectado a todas as
  janelas via ``reports`` (análogo ao *masked mean pool* da cross-attention).
- **Features do nó ``video``** — média das features tabulares por janela
  (demografia + prosódia + tipo de pergunta).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

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


def _chain_edges(num_nodes: int, device: torch.device) -> Tensor:
    """Arestas ``i → i+1`` para sequência temporal (0 arestas se T < 2)."""
    if num_nodes < 2:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    src = torch.arange(num_nodes - 1, device=device)
    dst = torch.arange(1, num_nodes, device=device)
    return torch.stack([src, dst])


def _same_index_edges(num_nodes: int, device: torch.device) -> Tensor:
    """Arestas ``i → i`` (alinhamento áudio-texto na mesma janela)."""
    idx = torch.arange(num_nodes, device=device)
    return torch.stack([idx, idx])


def _to_video_edges(num_nodes: int, device: torch.device) -> Tensor:
    """Arestas ``window_i → video_0`` (todas as janelas reportam ao nó de vídeo)."""
    src = torch.arange(num_nodes, device=device)
    dst = torch.zeros(num_nodes, dtype=torch.long, device=device)
    return torch.stack([src, dst])


def build_video_hetero_graph(
    audio_seq: Tensor,
    text_seq: Tensor,
    tab_seq: Tensor | None = None,
) -> Any:
    """Monta um ``HeteroData`` a partir das sequências de janelas de UM vídeo.

    Args:
        audio_seq: ``(T, d_audio)`` — embeddings de áudio por janela.
        text_seq: ``(T, d_text)`` — embeddings de texto por janela.
        tab_seq: ``(T, d_tab)`` opcional — features tabulares por janela.

    Returns:
        ``HeteroData`` pronto para ``Batch.from_data_list`` ou inferência unitária.
    """
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

    if tab_seq is not None and tab_seq.numel() > 0:
        data["video"].x = tab_seq.float().mean(dim=0, keepdim=True)
    else:
        data["video"].x = torch.zeros(1, 1, device=device)

    data["audio", "temporal", "audio"].edge_index = _chain_edges(t_windows, device)
    data["text", "temporal", "text"].edge_index = _chain_edges(t_windows, device)
    data["audio", "aligns", "text"].edge_index = _same_index_edges(t_windows, device)
    data["audio", "reports", "video"].edge_index = _to_video_edges(t_windows, device)
    data["text", "reports", "video"].edge_index = _to_video_edges(t_windows, device)

    return data


def collate_graph_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate que converte cada vídeo em grafo heterogêneo e empilha com PyG ``Batch``.

    Returns:
        ``{"graph": Batch, "label": (B, 1), "video_id": list[str]}``
    """
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
