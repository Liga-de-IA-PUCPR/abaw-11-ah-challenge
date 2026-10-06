"""Codificadores por modalidade (ramos "LEFT" da figura do plano MoE) — registry ``ENCODERS``.

Cada ramo do ``moe_fusion`` é declarado no YAML (``model.branches.<nome>``) com um
``encoder`` deste registry e a sua entrada:

- ``sequence`` — features POR JANELA de uma coluna do Parquet ``(B, T, d_in)`` → norm →
  projeção (ou :class:`~src.models.blocks.SoftMoE` sobre grupos, ex. recortes
  face/olhos/boca) → codificador temporal → pooling → ``(B, dim_out)``. Opcional: cabeça
  POR JANELA (``window_weight > 0``) supervisionada pelos rótulos de janela
  (``time_detailed_ah``). Ex.: áudio, rosto, tabular, ``text_emb`` congelado.
- ``vector`` — vetor POR VÍDEO (coluna replicada nas janelas, ex. ASR timing, marcadores
  de hesitação, cena) → norm → MLP → ``(B, dim)``.
- ``hf_text`` — transcrição tokenizada → ``AutoModel`` do HuggingFace com as primeiras
  camadas congeladas → CLS/média → ``(B, hidden)``. Ex.: RoBERTa-GoEmotions fine-tunado.

Trocar o embedder de uma modalidade = apontar ``column`` para outra coluna (gerada por
``mode=featurize_columns``) ou trocar ``model_name`` do ramo de texto; trocar a arquitetura
do ramo = mudar ``encoder``/``temporal``/``pool``/``mixer`` no YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from src.logger import get_logger
from src.models.blocks import (
    ProjectionBlock,
    SoftMoE,
    TemporalEncoder,
    build_norm,
    build_pool,
    first_valid,
)

log = get_logger("models.encoders")


class SequenceEncoder(nn.Module):
    """Sequência de janelas ``(B, T, d_in)`` → vetor do vídeo ``(B, out_dim)``.

    Args:
        in_dim: dimensão da coluna por janela.
        dim: largura interna (projeção / codificador temporal).
        norm: normalização de entrada (``batch`` | ``layer`` | ``none``).
        mixer: ``linear`` (projeção) ou ``moe`` (SoftMoE sobre ``groups`` blocos
            concatenados — ex. ``[face ‖ eyes ‖ mouth]`` com ``groups=3``).
        temporal: ``identity`` | ``gru`` | ``transformer``.
        pool: ``mean`` | ``max`` | ``attention`` | ``stats``.
        window_head: cria a cabeça por janela (logit de A/H em cada janela).
    """

    def __init__(
        self,
        in_dim: int,
        dim: int = 256,
        norm: str = "batch",
        mixer: str = "linear",
        groups: int = 1,
        num_experts: int = 3,
        temporal: str = "identity",
        num_layers: int = 1,
        num_heads: int = 4,
        pool: str = "attention",
        dropout: float = 0.1,
        window_head: bool = False,
    ) -> None:
        super().__init__()
        self.norm = build_norm(norm, in_dim)
        self.mixer = mixer
        if mixer == "moe":
            if in_dim % groups:
                raise ValueError(f"in_dim={in_dim} não divide em groups={groups}")
            self.moe = SoftMoE(in_dim, dim, num_experts=num_experts, dropout=dropout)
            self.post = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout))
        elif mixer == "linear":
            self.proj = ProjectionBlock(in_dim, dim, dropout)
        else:
            raise ValueError(f"mixer desconhecido: {mixer!r} (linear|moe)")
        self.temporal = TemporalEncoder(temporal, dim, num_layers, num_heads, dropout)
        self.pool, self.out_dim = build_pool(pool, dim)
        self.window_head = nn.Linear(dim, 1) if window_head else None

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``(h (B, out_dim), logits por janela (B, T) | None, pesos do mixer (B, T, K) | None)``"""
        x = self.norm(x, mask)
        mix_w = None
        if self.mixer == "moe":
            tokens, mix_w = self.moe(x)
            tokens = self.post(tokens)
        else:
            tokens = self.proj(x)
        tokens = self.temporal(tokens, mask)
        window_logits = (
            self.window_head(tokens).squeeze(-1) if self.window_head is not None else None
        )
        return self.pool(tokens, mask), window_logits, mix_w


class VectorEncoder(nn.Module):
    """Vetor do vídeo (coluna replicada nas janelas) → MLP ``(B, dim)``."""

    def __init__(self, in_dim: int, dim: int = 64, norm: str = "batch", dropout: float = 0.1):
        super().__init__()
        self.norm = build_norm(norm, in_dim)
        self.mlp = ProjectionBlock(in_dim, dim, dropout)
        self.out_dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None):
        v = first_valid(x, mask) if x.dim() == 3 else x
        return self.mlp(self.norm(v)), None, None


class HFTextEncoder(nn.Module):
    """Transcrição tokenizada → encoder HuggingFace (fine-tune parcial) → ``(B, hidden)``.

    Congela ``embeddings`` + as ``num_frozen_layers`` primeiras camadas (receita do plano:
    RoBERTa-GoEmotions com 4 camadas congeladas). ``trainable=False`` congela tudo
    (encoder fixo, ex. inicializado do ramo de texto de outro run via ``init_from``).
    """

    is_backbone = True  # parâmetros com lr próprio (lr_backbone) no otimizador

    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        num_frozen_layers: int = 4,
        pooling: str = "cls",
        trainable: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        log.info(f"Carregando encoder de texto (fine-tune): {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.out_dim = int(self.backbone.config.hidden_size)
        self._freeze(num_frozen_layers, trainable)
        # ``from_pretrained`` devolve o modelo em eval() e o Lightning preserva o modo dos
        # submódulos: sem isto o fine-tune roda SEM dropout no backbone (e o R-Drop zera).
        # Encoder fixo (trainable=False) fica em eval: features determinísticas.
        self.backbone.train(trainable)

    def _freeze(self, num_frozen_layers: int, trainable: bool) -> None:
        if not trainable:
            self.backbone.requires_grad_(False)
            return
        frozen = [self.backbone.embeddings]
        layers = getattr(getattr(self.backbone, "encoder", None), "layer", [])
        frozen += list(layers[: max(0, int(num_frozen_layers))])
        for module in frozen:
            module.requires_grad_(False)
        n_frozen = min(int(num_frozen_layers), len(layers))
        log.info(f"Texto: embeddings + {n_frozen}/{len(layers)} camadas congeladas.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0], None, None
        m = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0), None, None


# Registry {nome do encoder no YAML: classe}
ENCODERS: dict[str, Callable[..., nn.Module]] = {
    "sequence": SequenceEncoder,
    "vector": VectorEncoder,
    "hf_text": HFTextEncoder,
}

# Chaves do bloco do ramo que NÃO são hiperparâmetros do encoder.
_BRANCH_KEYS = {
    "encoder",
    "role",
    "column",
    "aux_weight",
    "window_weight",
    "init_from",
    "max_length",
}


def build_encoder(spec: dict[str, Any], in_dim: int | None) -> nn.Module:
    """Instancia o encoder do ramo ``spec`` (bloco ``model.branches.<nome>`` do YAML)."""
    kind = spec.get("encoder", "sequence")
    if kind not in ENCODERS:
        raise KeyError(f"encoder '{kind}' desconhecido. Disponíveis: {sorted(ENCODERS)}")
    kwargs = {k: v for k, v in spec.items() if k not in _BRANCH_KEYS}
    if kind == "sequence":
        kwargs["window_head"] = float(spec.get("window_weight", 0.0) or 0.0) > 0
    if kind != "hf_text":
        if not in_dim:
            raise ValueError(f"ramo '{spec.get('column')}': dimensão de entrada desconhecida")
        kwargs["in_dim"] = int(in_dim)
    return ENCODERS[kind](**kwargs)
