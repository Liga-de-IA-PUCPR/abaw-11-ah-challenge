"""Blocos neurais reutilizáveis do modelo MoE (``moe_fusion``) — sequências de janelas.

Contrato comum: ``x (B, T, D)`` + ``mask (B, T)`` com ``True`` = janela de padding (o
``key_padding_mask`` do :func:`~src.data.datasets.collate_sequences`). Cada bloco é
pequeno e configurável por string no YAML (``build_pool``/``TemporalEncoder``), então trocar
a agregação temporal ou o codificador de uma modalidade é só mudar a config.

- Agregação janela→vídeo: ``mean`` | ``max`` | ``attention`` (atenção-MIL gated, Ilse et al.
  2018 — mesma do ``cross_attention``) | ``stats`` (``[μ, σ, μΔ, σΔ]``, resumo temporal do
  RAS, 2º lugar).
- Codificador temporal: ``identity`` | ``gru`` (bidirecional) | ``transformer`` (pequeno,
  pré-LN, posição senoidal). Mamba fica de fora por decisão do plano (só se a cabeça
  temporal virar gargalo comprovado).
- :class:`SoftMoE`: roteamento denso (softmax sobre K experts MLP), a mesma formulação do
  ``SoftMoE`` da ``vision-toolbelt-liga``, mas devolvendo também os pesos do roteador
  (balanceamento + telemetria) e aceitando dimensões iniciais arbitrárias ``(..., D)``.
- :class:`ReliabilityGate`: ``g_m = σ(MLP[b; h_m])`` da fusão ancorada em texto.

Importado só pelo caminho lightning (registro lazy do ``moe_fusion``), então o ``torch`` no
topo não afeta o caminho sklearn.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# =============================================================================
# Pooling temporal mascarado (janela → vídeo)
# =============================================================================


def _valid(mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    """``(B, T, 1)`` float com 1 nas janelas reais (tudo 1 se ``mask`` é ``None``)."""
    if mask is None:
        return x.new_ones(x.shape[0], x.shape[1], 1)
    return (~mask).unsqueeze(-1).to(x.dtype)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Média sobre T ignorando o padding → ``(B, D)``."""
    valid = _valid(mask, x)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def masked_max(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Máximo sobre T ignorando o padding → ``(B, D)``."""
    if mask is not None:
        x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
    return x.max(dim=1).values


def first_valid(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Primeira janela real de cada vídeo → ``(B, D)`` (colunas de vídeo replicadas)."""
    if mask is None:
        return x[:, 0]
    idx = (~mask).float().argmax(dim=1)  # 1º índice não-padding
    return x[torch.arange(x.shape[0], device=x.device), idx]


class GatedAttentionPool(nn.Module):
    """Pooling atenção-MIL gated (Ilse et al. 2018): peso aprendido por janela, softmax
    mascarado sobre T e soma ponderada — janelas informativas dominam (vídeo positivo se
    ALGUMA janela tem A/H)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn_V = nn.Linear(dim, dim)
        self.attn_U = nn.Linear(dim, dim)
        self.attn_w = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        gate = torch.tanh(self.attn_V(x)) * torch.sigmoid(self.attn_U(x))
        scores = self.attn_w(gate).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        return (torch.softmax(scores, dim=1).unsqueeze(-1) * x).sum(dim=1)


class StatsPool(nn.Module):
    """``[μ, σ, μΔ, σΔ]`` mascarado (média/desvio do sinal e da 1ª diferença temporal)."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        valid = _valid(mask, x)
        mu = masked_mean(x, mask)
        sigma = (((x - mu.unsqueeze(1)) ** 2 * valid).sum(1) / valid.sum(1).clamp_min(1.0)).sqrt()
        dx = x[:, 1:] - x[:, :-1]
        dvalid = valid[:, 1:] * valid[:, :-1]  # par (t, t+1) só se ambas são reais
        n = dvalid.sum(1).clamp_min(1.0)
        dmu = (dx * dvalid).sum(1) / n
        dsigma = (((dx - dmu.unsqueeze(1)) ** 2 * dvalid).sum(1) / n).sqrt()
        return torch.cat([mu, sigma, dmu, dsigma], dim=-1)


class _FnPool(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.fn(x, mask)


def build_pool(kind: str, dim: int) -> tuple[nn.Module, int]:
    """Pooling ``kind`` sobre features de dimensão ``dim`` → ``(módulo, dim de saída)``."""
    if kind == "mean":
        return _FnPool(masked_mean), dim
    if kind == "max":
        return _FnPool(masked_max), dim
    if kind == "attention":
        return GatedAttentionPool(dim), dim
    if kind == "stats":
        return StatsPool(), 4 * dim
    raise ValueError(f"pool desconhecido: {kind!r} (mean|max|attention|stats)")


# =============================================================================
# Normalização / codificação temporal
# =============================================================================


class MaskedBatchNorm(nn.Module):
    """BatchNorm por feature calculado SÓ nas janelas reais (escalas heterogêneas: F0 ~500,
    contagens ~2, probabilidades ~0,5). Aceita ``(B, D)`` ou ``(B, T, D)``; com < 2 linhas
    válidas no treino usa as estatísticas acumuladas (evita o erro do BN com lote 1)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        valid = None if mask is None or x.dim() == 2 else (~mask).reshape(-1)
        rows = flat if valid is None else flat[valid]
        if rows.shape[0] < 2:
            normed = F.batch_norm(
                rows, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias,
                training=False, eps=self.bn.eps,
            )  # fmt: skip
        else:
            normed = self.bn(rows)
        if valid is None:
            return normed.reshape(shape)
        out = torch.zeros_like(flat)
        out[valid] = normed.to(out.dtype)
        return out.reshape(shape)


class _LayerNorm(nn.LayerNorm):
    """LayerNorm com a mesma assinatura ``(x, mask)`` dos demais blocos (ignora a máscara)."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x)


class _NoNorm(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x


def build_norm(kind: str | None, dim: int) -> nn.Module:
    """Normalização de entrada ``(x, mask)``: ``batch`` (por feature, só janelas reais) |
    ``layer`` (por amostra) | ``none``."""
    if kind == "batch":
        return MaskedBatchNorm(dim)
    if kind == "layer":
        return _LayerNorm(dim)
    if kind in ("none", None):
        return _NoNorm()
    raise ValueError(f"norm desconhecida: {kind!r} (batch|layer|none)")


def sinusoidal_positions(t: int, dim: int, device: torch.device) -> torch.Tensor:
    """Codificação posicional senoidal ``(T, D)`` (sem parâmetros; qualquer T)."""
    pos = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(t, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : dim // 2]
    return pe


class TemporalEncoder(nn.Module):
    """Contexto entre janelas: ``identity`` | ``gru`` (bidirecional) | ``transformer``."""

    def __init__(
        self, kind: str, dim: int, num_layers: int = 1, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.rnn = nn.GRU(
                dim, dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True
            )
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                dim, num_heads, dim_feedforward=2 * dim, dropout=dropout,
                batch_first=True, norm_first=True,
            )  # fmt: skip
            self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        elif kind != "identity":
            raise ValueError(f"temporal desconhecido: {kind!r} (identity|gru|transformer)")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.kind == "gru":
            lengths = (
                torch.full((x.shape[0],), x.shape[1]) if mask is None else (~mask).sum(1).cpu()
            )
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.clamp_min(1), batch_first=True, enforce_sorted=False
            )
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.shape[1]
            )
            return out
        if self.kind == "transformer":
            x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device).to(x.dtype)
            return self.encoder(x, src_key_padding_mask=mask)
        return x


# =============================================================================
# Mixture-of-Experts + fusão
# =============================================================================


class ExpertMLP(nn.Sequential):
    """Expert de 2 camadas (Linear → GELU → Dropout → Linear)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None, dropout: float = 0.0):
        hidden = hidden or in_dim
        super().__init__(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
        )


class SoftMoE(nn.Module):
    """Soft Mixture-of-Experts: ``w = softmax(router(x))``; saída ``Σ_k w_k · expert_k(x)``.

    Roteamento denso (todo expert roda em todo item) — simples de treinar, sem descarte de
    tokens. Aceita ``x (..., in_dim)``; devolve ``(saída (..., out_dim), pesos (..., K))``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int = 4,
        hidden: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts deve ser >= 1 (recebido {num_experts})")
        self.num_experts = num_experts
        self.router = nn.Linear(in_dim, num_experts)
        self.experts = nn.ModuleList(
            ExpertMLP(in_dim, out_dim, hidden, dropout) for _ in range(num_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.router(x), dim=-1)
        outs = torch.stack([e(x) for e in self.experts], dim=-2)  # (..., K, out_dim)
        return torch.einsum("...k,...kd->...d", weights, outs), weights


def load_balance_loss(weights: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """KL(uso ‖ uniforme) dos experts — aplicado SÓ se um expert passa de ``threshold`` do
    tráfego no lote (regra do plano: balancear quando o desbalanceio é medido, não antes).

    ``weights``: pesos do roteador ``(..., K)``. Devolve 0 quando o uso está equilibrado.
    """
    usage = weights.reshape(-1, weights.shape[-1]).mean(dim=0)  # (K,) — fração do tráfego
    if usage.max().item() <= threshold:
        return weights.new_zeros(())
    k = usage.shape[0]
    return (usage * (usage * k).clamp_min(1e-9).log()).sum()


class ProjectionBlock(nn.Sequential):
    """``d_m(·)``: Linear → LayerNorm → GELU → Dropout (leva cada modalidade a ``out_dim``)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
        )


class ReliabilityGate(nn.Module):
    """``g_m = σ(MLP[b; h_m])`` — quanto confiar na modalidade ``m`` dado o texto ``b``.

    ``kind="scalar"`` (1 peso por amostra) ou ``"vector"`` (1 peso por dimensão).
    ``init_bias`` < 0 começa com a porta quase fechada (o modelo parte do texto puro e abre
    as modalidades que ajudam).
    """

    def __init__(self, dim: int, kind: str = "scalar", init_bias: float = 0.0) -> None:
        super().__init__()
        out = 1 if kind == "scalar" else dim
        self.mlp = nn.Sequential(nn.Linear(2 * dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, out))
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, anchor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(torch.cat([anchor, h], dim=-1)))
