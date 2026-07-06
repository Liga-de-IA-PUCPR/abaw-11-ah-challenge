"""GNN heterogêneo para classificação A/H a nível de vídeo (BAH / ABAW11).

Substitui a pilha *local self-attention + cross-attention + pool* das branches
``matheus-local-attn`` e ``luiz`` por **message passing** explícito em grafo,
usando :class:`~gnn_modalblocks.architectures.hetero_gat.HeteroGAT` da lib
``gnn-modalblocks``.

Topologia do grafo (por vídeo):
  - nós ``audio`` e ``text`` = janelas temporais;
  - arestas ``temporal`` = vizinhança local intra-modal;
  - arestas ``aligns`` = acoplamento cross-modal por janela;
  - nó ``video`` = readout global (features tabulares agregadas).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.data.graph_builder import BAH_GRAPH_METADATA
from src.logger import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger("models.hetero_gnn")


def _build_gnn_module(
    dim_a: int,
    dim_b: int,
    dim_tab: int,
    hidden_channels: int,
    heads: int,
    out_channels: int,
    dropout: float,
):
    """Instancia o ``HeteroGAT`` + cabeça de classificação (imports lazy)."""
    import torch
    from torch import nn

    from gnn_modalblocks import ENCODERS

    in_channels = {
        "audio": dim_a,
        "text": dim_b,
        "video": max(dim_tab, 1),
    }

    gat = ENCODERS.build(
        "hetero_gat",
        metadata=BAH_GRAPH_METADATA,
        in_channels_dict=in_channels,
        hidden_channels=hidden_channels,
        heads=heads,
        out_channels=out_channels,
        head_node_type="video",
        num_classes=1,
        task="binary",
    )

    class _BahVideoHeteroGNN(nn.Module):
        """Wrapper que devolve logits (BCEWithLogits) em vez de probabilidades."""

        def __init__(self) -> None:
            super().__init__()
            self.gat = gat
            self.dropout = nn.Dropout(dropout)
            self.refine = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_channels, 1),
            )

        def forward(self, graph) -> torch.Tensor:
            _proba, z_dict = self.gat(graph)
            video_z = z_dict.get("video")
            if video_z is None or video_z.size(0) == 0:
                return torch.zeros(0, 1, device=next(self.parameters()).device)
            return self.refine(self.dropout(video_z))

    return _BahVideoHeteroGNN()


class HeteroGnnFusion:
    """Fachada registrável (family=lightning) para o GNN heterogêneo."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.dim_a = int(cfg.get("dim_a", 768))
        self.dim_b = int(cfg.get("dim_b", 768))
        self.dim_tab = int(cfg.get("dim_tab", 32))
        self.hidden_channels = int(cfg.get("hidden_channels", 128))
        self.heads = int(cfg.get("heads", 4))
        self.out_channels = int(cfg.get("out_channels", 64))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))

    @classmethod
    def from_config(cls, config: Any) -> HeteroGnnFusion:
        return cls(cfg=config)

    def build_module(self):
        return _build_gnn_module(
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            dim_tab=self.dim_tab,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            out_channels=self.out_channels,
            dropout=self.dropout,
        )

    def build_lightning_module(self):
        return _build_lit_module(
            fusion=self.build_module(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def load_from_checkpoint(self, ckpt_path, map_location=None):
        import torch

        ckpt = torch.load(str(ckpt_path), map_location=map_location)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        module = self.build_module()
        prefix = "model."
        fusion_state = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        module.load_state_dict(fusion_state or state_dict)
        module.eval()
        return module


def _build_lit_module(fusion, lr: float, weight_decay: float):
    import lightning as L
    import torch
    from torch import nn, optim
    from torchmetrics.classification import BinaryAveragePrecision, MulticlassF1Score

    class _ClassificationMetrics(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.val_ap = BinaryAveragePrecision()
            self.val_f1_macro = MulticlassF1Score(num_classes=2, average="macro")
            self.test_ap = BinaryAveragePrecision()
            self.test_f1_macro = MulticlassF1Score(num_classes=2, average="macro")

    class LitHeteroGnn(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = fusion
            self.metrics = _ClassificationMetrics()
            self._lr = lr
            self._weight_decay = weight_decay

        def _shared_step(self, batch):
            logit = self.model(batch["graph"])
            label = batch["label"].float()
            loss = nn.functional.binary_cross_entropy_with_logits(logit, label)
            proba = torch.sigmoid(logit)
            return loss, proba, label.int()

        def training_step(self, batch, batch_idx):
            loss, _, _ = self._shared_step(batch)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            pred = (proba >= 0.5).int().squeeze(-1)
            target = label.squeeze(-1)
            self.metrics.val_ap.update(proba.squeeze(-1), target)
            self.metrics.val_f1_macro.update(pred, target)

        def on_validation_epoch_end(self) -> None:
            self.log("val_ap", self.metrics.val_ap, prog_bar=True)
            self.log("val_f1_macro", self.metrics.val_f1_macro, prog_bar=True)
            self.metrics.val_ap.reset()
            self.metrics.val_f1_macro.reset()

        def test_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            pred = (proba >= 0.5).int().squeeze(-1)
            target = label.squeeze(-1)
            self.metrics.test_ap.update(proba.squeeze(-1), target)
            self.metrics.test_f1_macro.update(pred, target)

        def on_test_epoch_end(self) -> None:
            self.log("test_ap", self.metrics.test_ap)
            self.log("test_f1_macro", self.metrics.test_f1_macro)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            _, proba, _ = self._shared_step(batch)
            return {"video_ids": batch["video_id"], "proba": proba.squeeze(-1)}

        def configure_optimizers(self):
            return optim.AdamW(
                self.parameters(),
                lr=self._lr,
                weight_decay=self._weight_decay,
            )

    return LitHeteroGnn()
