"""Pré-treino não supervisionado com HeteroGAE (gnn-modalblocks)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.data.graph_builder import BAH_GRAPH_METADATA, build_video_hetero_graph
from src.logger import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger("models.hetero_gae_pretrain")


def _build_gae_module(dim_a: int, dim_b: int, dim_tab: int, hidden_channels: int, out_channels: int):
    import torch
    from torch import nn
    from torch_geometric.data import Batch

    from gnn_modalblocks import ENCODERS

    in_channels = {
        "audio": dim_a,
        "text": dim_b,
        "video": max(dim_tab, 1),
    }

    gae = ENCODERS.build(
        "hetero_gae",
        metadata=BAH_GRAPH_METADATA,
        in_channels_dict=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        anchor_node_type="video",
    )

    class _BahVideoGAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gae = gae

        def forward(
            self,
            feat_a: torch.Tensor,
            feat_b: torch.Tensor,
            tab_seq: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            graphs = []
            for i in range(feat_a.size(0)):
                t = int(lengths[i].item())
                graphs.append(
                    build_video_hetero_graph(
                        feat_a[i, :t],
                        feat_b[i, :t],
                        tab_seq[i, :t],
                    )
                )
            batch_graph = Batch.from_data_list(graphs)
            z_dict = self.gae.encode(batch_graph)
            return self.gae.recon_loss(z_dict, batch_graph)

    return _BahVideoGAE()


class HeteroGaePretrain:
    """Fachada para pré-treino GAE (family=lightning via GaePretrainTrainer)."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.dim_a = int(cfg.get("dim_a", 768))
        self.dim_b = int(cfg.get("dim_b", 768))
        self.dim_tab = int(cfg.get("dim_tab", 32))
        self.hidden_channels = int(cfg.get("hidden_channels", 128))
        self.out_channels = int(cfg.get("out_channels", 64))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))

    @classmethod
    def from_config(cls, config: Any) -> HeteroGaePretrain:
        return cls(cfg=config)

    def build_module(self):
        return _build_gae_module(
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            dim_tab=self.dim_tab,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
        )

    def build_lightning_module(self):
        return _build_gae_lit_module(
            gae=self.build_module(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def save_encoder(self, lit_module, out_path) -> None:
        import torch

        encoder_state = lit_module.model.gae.encoder.state_dict()
        torch.save({"encoder": encoder_state}, out_path)
        log.info(f"Encoder GAE salvo em: {out_path}")


def _build_gae_lit_module(gae, lr: float, weight_decay: float):
    import lightning as L
    from torch import optim

    class LitHeteroGAE(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = gae
            self._lr = lr
            self._weight_decay = weight_decay

        def _shared_step(self, batch):
            return self.model(
                batch["audio_seq"],
                batch["text_seq"],
                batch["tab_seq"],
                batch["lengths"],
            )

        def training_step(self, batch, batch_idx):
            loss = self._shared_step(batch)
            self.log("train_recon_loss", loss, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            loss = self._shared_step(batch)
            self.log("val_recon_loss", loss, on_epoch=True, prog_bar=True)

        def configure_optimizers(self):
            return optim.AdamW(
                self.parameters(), lr=self._lr, weight_decay=self._weight_decay
            )

    return LitHeteroGAE()
