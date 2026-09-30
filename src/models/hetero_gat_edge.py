"""HeteroGAT local com ``edge_attr`` (pesos de aresta) — não altera gnn-modalblocks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

from gnn_modalblocks.architectures.hetero_utils import (
    add_reverse_edge_types,
    augment_edge_index_dict,
)


def _make_hetero_conv(
    edge_types: list[tuple[str, str, str]],
    in_channels: int,
    out_channels: int,
    heads: int,
    edge_dim: int | None,
) -> HeteroConv:
    return HeteroConv(
        {
            et: GATConv(
                in_channels,
                out_channels,
                heads=heads,
                concat=heads > 1,
                add_self_loops=False,
                edge_dim=edge_dim,
            )
            for et in edge_types
        },
        aggr="sum",
    )


class HeteroGATEdgeAttr(nn.Module):
    """Igual ao HeteroGAT do gnn-modalblocks, mas propaga ``edge_attr``."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels_dict: dict[str, int],
        hidden_channels: int = 64,
        heads: int = 4,
        out_channels: int = 32,
        head_node_type: str = "video",
        num_classes: int = 1,
        task: str = "binary",
        num_layers: int = 2,
        edge_dim: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers deve ser >= 1, recebeu {num_layers}")

        node_types, edge_types = metadata
        full_et = add_reverse_edge_types(edge_types)
        self._edge_dim = int(edge_dim) if edge_dim else None

        self.projections = nn.ModuleDict(
            {nt: nn.Linear(in_channels_dict[nt], hidden_channels) for nt in node_types}
        )

        convs: list[HeteroConv] = []
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            if is_last:
                convs.append(
                    _make_hetero_conv(
                        full_et, hidden_channels, out_channels, heads=1, edge_dim=self._edge_dim
                    )
                )
            else:
                convs.append(
                    _make_hetero_conv(
                        full_et,
                        hidden_channels,
                        hidden_channels // heads,
                        heads=heads,
                        edge_dim=self._edge_dim,
                    )
                )
        self.convs = nn.ModuleList(convs)
        out_dim = num_classes if task == "multiclass" else 1
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_dim),
        )
        self._node_types = node_types
        self.head_node_type = head_node_type
        self.task = task

    def forward(self, data: HeteroData) -> tuple[Tensor, dict[str, Tensor]]:
        x_dict: dict[str, Tensor] = {}
        dev = next(self.parameters()).device
        for nt in self._node_types:
            if hasattr(data[nt], "x") and data[nt].x.size(0) > 0:
                x_dict[nt] = self.projections[nt](data[nt].x.float())
            else:
                x_dict[nt] = torch.zeros(0, self.projections[nt].out_features, device=dev)

        ei_dict: dict[tuple[str, str, str], Tensor] = {}
        ea_dict: dict[tuple[str, str, str], Tensor] = {}
        for et in data.edge_types:
            store = data[et]
            if hasattr(store, "edge_index"):
                ei_dict[et] = store.edge_index
                if self._edge_dim and hasattr(store, "edge_attr") and store.edge_attr is not None:
                    ea_dict[et] = store.edge_attr.float()
                elif self._edge_dim:
                    n_e = int(store.edge_index.size(1))
                    ea_dict[et] = torch.ones(n_e, self._edge_dim, device=dev)

        aug_ei = augment_edge_index_dict(ei_dict)
        aug_ea = dict(ea_dict)
        for et, ea in list(ea_dict.items()):
            src, rel, dst = et
            rev = (dst, f"rev_{rel}", src)
            if rev not in aug_ea and src != dst:
                aug_ea[rev] = ea

        h = x_dict
        z_dict = h
        for layer_idx, conv in enumerate(self.convs):
            if self._edge_dim:
                h = conv(h, aug_ei, aug_ea)
            else:
                h = conv(h, aug_ei)
            if layer_idx < len(self.convs) - 1:
                h = {k: F.elu(v) for k, v in h.items()}
            z_dict = h

        head_z = z_dict.get(self.head_node_type)
        if head_z is not None and head_z.size(0) > 0:
            logits = self.classifier(head_z)
            scores = torch.sigmoid(logits) if self.task == "binary" else logits
        else:
            scores = torch.zeros(0, 1, device=dev)
        return scores, z_dict


def build_ca_edge_scorer(
    dim_a: int,
    dim_b: int,
    dim_tab: int,
    common_dim: int,
    num_heads: int,
    dropout: float,
    use_tabular: bool = True,
) -> Any:
    """Cross-attention Luiz → matriz de alinhamento + pesos MIL para ``reports``."""
    from torch import nn

    fuse_tab = bool(use_tabular) and dim_tab > 0

    class _CaEdgeScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fuse_tab = fuse_tab
            self.proj_a = nn.Linear(dim_a, common_dim)
            self.proj_b = nn.Linear(dim_b, common_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(common_dim)
            if fuse_tab:
                self.tab_in_norm = nn.BatchNorm1d(dim_tab)
                self.proj_tab = nn.Linear(dim_tab, common_dim)
                self.tab_norm = nn.LayerNorm(common_dim)
                self.token_fuse = nn.Linear(common_dim * 2, common_dim)
                self.token_norm = nn.LayerNorm(common_dim)
            self.attn_V = nn.Linear(common_dim, common_dim)
            self.attn_U = nn.Linear(common_dim, common_dim)
            self.attn_w = nn.Linear(common_dim, 1)

        def _tab_tokens(self, feat_tab: Tensor, mask: Tensor | None) -> Tensor:
            b, t, d = feat_tab.shape
            flat = feat_tab.reshape(b * t, d)
            if mask is not None:
                valid = (~mask).reshape(b * t)
                normed = flat.clone()
                if valid.any():
                    normed[valid] = self.tab_in_norm(flat[valid])
            else:
                normed = self.tab_in_norm(flat)
            tok = self.tab_norm(torch.relu(self.proj_tab(normed)))
            return tok.reshape(b, t, -1)

        def forward(
            self,
            feat_a: Tensor,
            feat_b: Tensor,
            feat_tab: Tensor | None = None,
            key_padding_mask: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor]:
            """Returns ``align_attn (B,T,T)``, ``report_w (B,T)``, ``fused (B,T,C)``."""
            q = self.proj_a(feat_a)
            kv = self.proj_b(feat_b)
            attn_out, attn_w = self.cross_attn(
                q,
                kv,
                kv,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            fused = self.norm(q + attn_out)
            if self.fuse_tab:
                if feat_tab is None:
                    raise ValueError("CA edge scorer com tabular requer feat_tab")
                tab_tok = self._tab_tokens(feat_tab, key_padding_mask)
                fused = self.token_norm(self.token_fuse(torch.cat([fused, tab_tok], dim=-1)))

            gate = torch.tanh(self.attn_V(fused)) * torch.sigmoid(self.attn_U(fused))
            scores = self.attn_w(gate).squeeze(-1)
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask, float("-inf"))
            report_w = torch.softmax(scores, dim=1)
            # Softmax sobre -inf → nan; substitui padding por 0.
            if key_padding_mask is not None:
                report_w = report_w.masked_fill(key_padding_mask, 0.0)
                report_w = report_w / report_w.sum(dim=1, keepdim=True).clamp_min(1e-8)

            # Mascara colunas padding na atenção áudio→texto.
            align = attn_w
            if key_padding_mask is not None:
                align = align.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
                align = align.masked_fill(key_padding_mask.unsqueeze(2), 0.0)
            return align, report_w, fused

    return _CaEdgeScorer()
