"""Encoder de features tabulares de suporte (hesitação + texto) — late fusion sem esmagamento.

Porta as ideias do protocolo Luiz (``tab_fusion=token`` + ``pool=attention``) para modelos
que consomem ``tab_seq (B, T, dim_tab)`` em paralelo aos embeddings deep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


def build_tab_support_encoder(
    dim_tab: int,
    out_dim: int,
    *,
    pool: str = "attention",
    dropout: float = 0.1,
):
    """Factory lazy: BN per-feature → proj → pooling temporal mascarado."""
    import torch
    from torch import nn

    pool = str(pool).lower()

    class _TabSupportEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = pool
            self.tab_in_norm = nn.BatchNorm1d(dim_tab)
            self.proj_tab = nn.Linear(dim_tab, out_dim)
            self.tab_norm = nn.LayerNorm(out_dim)
            self.drop = nn.Dropout(dropout)
            if pool == "attention":
                self.attn_V = nn.Linear(out_dim, out_dim)
                self.attn_U = nn.Linear(out_dim, out_dim)
                self.attn_w = nn.Linear(out_dim, 1)

        def _tab_tokens(self, feat_tab: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
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

        def _masked_attention_pool(
            self, x: torch.Tensor, mask: torch.Tensor | None
        ) -> torch.Tensor:
            gate = torch.tanh(self.attn_V(x)) * torch.sigmoid(self.attn_U(x))
            scores = self.attn_w(gate).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (weights * x).sum(dim=1)

        @staticmethod
        def _masked_max(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1).values

        @staticmethod
        def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            if mask is None:
                return x.mean(dim=1)
            valid = (~mask).unsqueeze(-1).float()
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        def forward(
            self,
            feat_tab: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """``(B, T, dim_tab)`` → ``(B, out_dim)``."""
            tokens = self._tab_tokens(feat_tab, key_padding_mask)
            tokens = self.drop(tokens)
            if self.pool == "attention":
                return self._masked_attention_pool(tokens, key_padding_mask)
            if self.pool == "max":
                return self._masked_max(tokens, key_padding_mask)
            return self._masked_mean(tokens, key_padding_mask)

    return _TabSupportEncoder()
