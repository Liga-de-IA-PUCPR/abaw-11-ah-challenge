"""MoERouter — sucessor do meta-router CA⊕GNN (painel RIGHT da figura do plano MoE).

Os "experts" são os MEMBROS já treinados (MoEFusionHead, texto/áudio/ASR unimodais, CA/GNN):
um roteador linear lê, por vídeo, os vetores pré-logit dos membros (``embeddings.npy``,
projetados) e os seus logits, e devolve pesos ``softmax(K)`` sobre os membros; a predição é
a combinação dos logits com esses pesos, com limiar FIXO ``τ = 0.5``::

    w(x) = softmax(Linear([proj_1(e_1) ‖ … ‖ proj_K(e_K) ‖ logit_1 … logit_K]))
    p(x) = σ(Σ_k w_k(x) · logit_k(x))

O router antigo só via probabilidades escalares; aqui ele "vê features" — a diferença que
motiva o MoE no plano. Membros sem embedding entram só pelo logit. Treino full-batch
(AdamW + BCE) com early stopping num holdout interno por participante.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.logger import get_logger

log = get_logger("models.moe_router")


@dataclass
class RouterInputs:
    """Entradas de K membros p/ N vídeos: logits ``(N, K)`` + embeddings por membro."""

    logits: np.ndarray
    embeddings: list[np.ndarray | None]

    def take(self, idx: np.ndarray) -> RouterInputs:
        return RouterInputs(
            self.logits[idx], [None if e is None else e[idx] for e in self.embeddings]
        )


class MoERouterNet(nn.Module):
    """Roteador sobre membros: projeções dos embeddings + logits → pesos ``(N, K)``."""

    def __init__(self, emb_dims: list[int], proj_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, proj_dim), nn.GELU())
            if d
            else nn.Identity()
            for d in emb_dims
        )
        self.has_emb = [bool(d) for d in emb_dims]
        in_dim = proj_dim * sum(self.has_emb) + len(emb_dims)
        self.drop = nn.Dropout(dropout)
        self.router = nn.Linear(in_dim, len(emb_dims))

    def forward(
        self, logits: torch.Tensor, embeddings: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [p(e) for p, e, has in zip(self.proj, embeddings, self.has_emb, strict=True) if has]
        weights = torch.softmax(self.router(self.drop(torch.cat([*feats, logits], dim=-1))), dim=-1)
        return (weights * logits).sum(dim=-1), weights


@dataclass
class MoERouter:
    """Treina/aplica o :class:`MoERouterNet` (full-batch; early stopping no holdout interno)."""

    proj_dim: int = 16
    dropout: float = 0.1
    lr: float = 1e-2
    weight_decay: float = 1e-2
    max_epochs: int = 500
    patience: int = 50
    seed: int = 42
    net: MoERouterNet | None = None

    def fit(
        self, train: RouterInputs, y: np.ndarray, val: RouterInputs, y_val: np.ndarray
    ) -> MoERouter:
        torch.manual_seed(self.seed)
        dims = [0 if e is None else e.shape[1] for e in train.embeddings]
        self.net = MoERouterNet(dims, self.proj_dim, self.dropout)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        xt, yt = _tensors(train), torch.as_tensor(y, dtype=torch.float32)
        xv, yv = _tensors(val), torch.as_tensor(y_val, dtype=torch.float32)
        best, best_state, wait = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            opt.zero_grad()
            logit, _ = self.net(*xt)
            nn.functional.binary_cross_entropy_with_logits(logit, yt).backward()
            opt.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = nn.functional.binary_cross_entropy_with_logits(
                    self.net(*xv)[0], yv
                ).item()
            if val_loss < best - 1e-5:
                best, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, inputs: RouterInputs) -> tuple[np.ndarray, np.ndarray]:
        """``(proba (N,), pesos dos membros (N, K))``."""
        if self.net is None:
            raise RuntimeError("MoERouter não treinado: chame fit() antes.")
        self.net.eval()
        with torch.no_grad():
            logit, weights = self.net(*_tensors(inputs))
        return torch.sigmoid(logit).numpy(), weights.numpy()


def to_logit(proba: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p)).astype(np.float32)


def _tensors(x: RouterInputs) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    embs = [None if e is None else torch.as_tensor(e, dtype=torch.float32) for e in x.embeddings]
    return torch.as_tensor(x.logits, dtype=torch.float32), embs
