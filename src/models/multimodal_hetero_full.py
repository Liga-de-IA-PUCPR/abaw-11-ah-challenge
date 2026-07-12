"""Modelo multimodal completo para competição — stack HeteroGAT + LatentGCN + contrastive."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.data.graph_builder import BAH_FULL_GRAPH_METADATA, build_video_hetero_graph
from src.logger import get_logger
from src.models.hetero_gnn import load_gae_projections

if TYPE_CHECKING:
    import torch

log = get_logger("models.multimodal_hetero_full")


def _build_full_module(
    dim_a: int,
    dim_b: int,
    dim_tab: int,
    common_dim: int,
    hidden_channels: int,
    heads: int,
    out_channels: int,
    gcn_out_dim: int,
    top_k: int,
    fusion: str,
    dropout: float,
    use_latent_gcn: bool = True,
    use_bilstm: bool = True,
    lstm_hidden: int = 64,
):
    import torch
    from torch import nn
    from torch_geometric.data import Batch

    from gnn_modalblocks import ENCODERS, MultimodalBlock

    in_channels = {
        "audio": dim_a,
        "text": dim_b,
        "fused": common_dim,
        "video": max(dim_tab, 1),
    }

    gat = ENCODERS.build(
        "hetero_gat",
        metadata=BAH_FULL_GRAPH_METADATA,
        in_channels_dict=in_channels,
        hidden_channels=hidden_channels,
        heads=heads,
        out_channels=out_channels,
        head_node_type="video",
        num_classes=1,
        task="binary",
    )

    class _MultimodalHeteroFullModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_a = nn.Linear(dim_a, common_dim)
            self.proj_b = nn.Linear(dim_b, common_dim)
            self.multimodal = MultimodalBlock(
                modality_dims={"audio": common_dim, "text": common_dim},
                fusion=fusion,
                out_dim=common_dim,
            )
            self.gat = gat
            self.dropout = nn.Dropout(dropout)
            self.use_latent_gcn = use_latent_gcn
            self.use_bilstm = use_bilstm

            if use_latent_gcn:
                self.latent_gcn = ENCODERS.build(
                    "latent_gcn",
                    in_channels=common_dim,
                    out_channels=gcn_out_dim,
                    num_nodes=64,
                    top_k=top_k,
                )
            else:
                self.latent_gcn = None
                gcn_out_dim_local = 0

            if use_bilstm:
                self.temporal = ENCODERS.build(
                    "temporal",
                    input_dim=common_dim,
                    hidden_dim=lstm_hidden,
                    num_layers=2,
                    dropout=dropout,
                )
                lstm_out = lstm_hidden
            else:
                self.temporal = None
                lstm_out = 0

            gcn_out = gcn_out_dim if use_latent_gcn else 0
            readout_dim = out_channels + gcn_out + lstm_out
            self.readout_dim = readout_dim
            self.classifier = nn.Sequential(
                nn.Linear(readout_dim, readout_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(readout_dim, 1),
            )

        def _fuse_windows(
            self,
            feat_a: torch.Tensor,
            feat_b: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Projeta e funde janelas: retorna (fused B,T,D), proj_a, proj_b)."""
            a = self.proj_a(feat_a)
            b = self.proj_b(feat_b)
            batch_size, t_max, channels = a.shape
            fused = self.multimodal(
                {
                    "audio": a.reshape(batch_size * t_max, channels),
                    "text": b.reshape(batch_size * t_max, channels),
                }
            ).view(batch_size, t_max, -1)
            return fused, a, b

        def _gat_video_z(
            self,
            feat_a: torch.Tensor,
            feat_b: torch.Tensor,
            fused: torch.Tensor,
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
                        fused_seq=fused[i, :t],
                    )
                )
            batch_graph = Batch.from_data_list(graphs)
            _scores, z_dict = self.gat(batch_graph)
            video_z = z_dict.get("video")
            if video_z is None or video_z.size(0) == 0:
                return torch.zeros(feat_a.size(0), out_channels, device=feat_a.device)
            return video_z

        def _latent_gcn_pool(
            self,
            fused: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            pooled: list[torch.Tensor] = []
            for i in range(fused.size(0)):
                t = int(lengths[i].item())
                nodes = fused[i, :t]
                if nodes.size(0) == 0:
                    pooled.append(torch.zeros(gcn_out_dim, device=fused.device))
                    continue
                h = self.latent_gcn(nodes)
                pooled.append(h.mean(dim=0))
            return torch.stack(pooled)

        def _bilstm_pool(
            self,
            fused: torch.Tensor,
            lengths: torch.Tensor,
        ) -> torch.Tensor:
            return self.temporal(fused, lengths)

        def _collect_align_pairs(
            self,
            proj_a: torch.Tensor,
            proj_b: torch.Tensor,
            lengths: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            pairs_a: list[torch.Tensor] = []
            pairs_b: list[torch.Tensor] = []
            for i in range(proj_a.size(0)):
                t = int(lengths[i].item())
                if t > 0:
                    pairs_a.append(proj_a[i, :t])
                    pairs_b.append(proj_b[i, :t])
            if not pairs_a:
                empty = proj_a.new_zeros(0, proj_a.size(-1))
                return empty, empty
            return torch.cat(pairs_a, dim=0), torch.cat(pairs_b, dim=0)

        def forward(
            self,
            feat_a: torch.Tensor,
            feat_b: torch.Tensor,
            tab_seq: torch.Tensor,
            lengths: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Retorna ``(logit, video_emb, align_a, align_b)``."""
            fused, proj_a, proj_b = self._fuse_windows(feat_a, feat_b)
            parts = [self._gat_video_z(feat_a, feat_b, fused, tab_seq, lengths)]

            if self.use_latent_gcn and self.latent_gcn is not None:
                parts.append(self._latent_gcn_pool(fused, lengths))
            if self.use_bilstm and self.temporal is not None:
                parts.append(self._bilstm_pool(fused, lengths))

            video_emb = self.dropout(torch.cat(parts, dim=-1))
            logit = self.classifier(video_emb)
            align_a, align_b = self._collect_align_pairs(proj_a, proj_b, lengths)
            return logit, video_emb, align_a, align_b

    return _MultimodalHeteroFullModule()


class MultimodalHeteroFullFusion:
    """Fachada registrável — modelo completo para competição."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.dim_a = int(cfg.get("dim_a", 768))
        self.dim_b = int(cfg.get("dim_b", 768))
        self.dim_tab = int(cfg.get("dim_tab", 32))
        self.common_dim = int(cfg.get("common_dim", 512))
        self.hidden_channels = int(cfg.get("hidden_channels", 128))
        self.heads = int(cfg.get("heads", 4))
        self.out_channels = int(cfg.get("out_channels", 64))
        self.gcn_out_dim = int(cfg.get("gcn_out_dim", 128))
        self.top_k = int(cfg.get("top_k", 8))
        self.fusion = str(cfg.get("fusion", "attention"))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.use_latent_gcn = bool(cfg.get("use_latent_gcn", True))
        self.use_bilstm = bool(cfg.get("use_bilstm", True))
        self.lstm_hidden = int(cfg.get("lstm_hidden", 64))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))
        self.pos_weight = cfg.get("pos_weight", "auto")
        self.gae_init = cfg.get("gae_init", None)
        c = cfg.get("contrastive", {}) or {}
        self.contrastive = {
            "enabled": bool(c.get("enabled", True)),
            "lambda_supcon": float(c.get("lambda_supcon", 0.1)),
            "lambda_triplet": float(c.get("lambda_triplet", 0.05)),
            "lambda_ntxent": float(c.get("lambda_ntxent", 0.05)),
            "temperature": float(c.get("temperature", 0.07)),
            "margin": float(c.get("margin", 0.2)),
            "miner": str(c.get("miner", "batch_hard")),
        }
        loss = cfg.get("loss", {}) or {}
        self.loss = {
            "type": str(loss.get("type", "bce")),
            "gamma": float(loss.get("gamma", 2.0)),
            "alpha": loss.get("alpha", 0.25),
        }
        self._resolved_pos_weight: float | None = None
        self._gae_init_path: str | None = None

    @classmethod
    def from_config(cls, config: Any) -> MultimodalHeteroFullFusion:
        return cls(cfg=config)

    def build_module(self):
        module = _build_full_module(
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            dim_tab=self.dim_tab,
            common_dim=self.common_dim,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            out_channels=self.out_channels,
            gcn_out_dim=self.gcn_out_dim,
            top_k=self.top_k,
            fusion=self.fusion,
            dropout=self.dropout,
            use_latent_gcn=self.use_latent_gcn,
            use_bilstm=self.use_bilstm,
            lstm_hidden=self.lstm_hidden,
        )
        gae_path = self._gae_init_path or self.gae_init
        if gae_path:
            load_gae_projections(module, str(gae_path))
        return module

    def build_lightning_module(self):
        return build_full_lit_module(
            fusion=self.build_module(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            pos_weight=self._resolved_pos_weight,
            contrastive_cfg=self.contrastive,
            loss_cfg=self.loss,
        )

    def load_from_checkpoint(self, ckpt_path, map_location=None):
        import torch

        ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        module = self.build_module()
        prefix = "model."
        fusion_state = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        module.load_state_dict(fusion_state or state_dict)
        module.eval()
        return module


def build_full_lit_module(
    fusion,
    lr: float,
    weight_decay: float,
    pos_weight: float | None = None,
    contrastive_cfg: dict[str, Any] | None = None,
    loss_cfg: dict[str, Any] | None = None,
):
    """LightningModule com BCE/focal + SupCon + triplet + NT-Xent cross-modal."""
    import lightning as L
    import torch

    from src.models.lightning_utils import (
        bce_with_logits,
        build_classification_metrics,
        configure_adamw_scheduler,
        focal_loss_with_logits,
        log_val_metrics,
    )

    lcfg = loss_cfg or {}
    loss_type = str(lcfg.get("type", "bce")).lower()
    focal_gamma = float(lcfg.get("gamma", 2.0))
    focal_alpha = lcfg.get("alpha", 0.25)
    focal_alpha = None if focal_alpha in (None, "none", "null") else float(focal_alpha)

    ccfg = contrastive_cfg or {}
    use_contrastive = bool(ccfg.get("enabled", False))
    lambda_supcon = float(ccfg.get("lambda_supcon", 0.1))
    lambda_triplet = float(ccfg.get("lambda_triplet", 0.05))
    lambda_ntxent = float(ccfg.get("lambda_ntxent", 0.05))
    temperature = float(ccfg.get("temperature", 0.07))
    margin = float(ccfg.get("margin", 0.2))
    miner_key = str(ccfg.get("miner", "batch_hard"))

    supcon_fn = triplet_fn = ntxent_fn = miner = None
    if use_contrastive:
        from gnn_modalblocks.contrastive.registry import LOSSES, MINERS

        supcon_fn = LOSSES.build("supervised_contrastive", temperature=temperature)
        ntxent_fn = LOSSES.build("nt_xent", temperature=temperature)
        if lambda_triplet > 0:
            triplet_fn = LOSSES.build("triplet_margin", margin=margin)
            miner = MINERS.build(miner_key)

    class LitMultimodalHeteroFull(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = fusion
            self.metrics = build_classification_metrics()
            self._lr = lr
            self._weight_decay = weight_decay
            self._pos_weight = pos_weight
            self._use_contrastive = use_contrastive
            self._lambda_supcon = lambda_supcon
            self._lambda_triplet = lambda_triplet
            self._lambda_ntxent = lambda_ntxent
            self.supcon_fn = supcon_fn
            self.triplet_fn = triplet_fn
            self.ntxent_fn = ntxent_fn
            self.miner = miner
            self._loss_type = loss_type
            self._focal_gamma = focal_gamma
            self._focal_alpha = focal_alpha

        def _forward(self, batch):
            return self.model(
                batch["audio_seq"],
                batch["text_seq"],
                batch["tab_seq"],
                batch["lengths"],
                key_padding_mask=batch["key_padding_mask"],
            )

        def _contrastive_losses(
            self,
            video_emb: torch.Tensor,
            align_a: torch.Tensor,
            align_b: torch.Tensor,
            label: torch.Tensor,
            log_aux: bool,
        ) -> torch.Tensor:
            extra = video_emb.new_tensor(0.0)
            if not self._use_contrastive:
                return extra

            if self.supcon_fn is not None and self._lambda_supcon > 0:
                supcon = self.supcon_fn(video_emb, label.squeeze(-1).long())
                extra = extra + self._lambda_supcon * supcon
                if log_aux:
                    self.log("train_supcon" if self.training else "val_supcon", supcon)

            if (
                self._lambda_triplet > 0
                and self.triplet_fn is not None
                and self.miner is not None
            ):
                a, p, n = self.miner(video_emb, label.squeeze(-1).long())
                trip = self.triplet_fn(a, p, n)
                extra = extra + self._lambda_triplet * trip
                if log_aux:
                    self.log("train_triplet" if self.training else "val_triplet", trip)

            if (
                self._lambda_ntxent > 0
                and self.ntxent_fn is not None
                and align_a.size(0) >= 2
            ):
                ntx = self.ntxent_fn(align_a, align_b)
                extra = extra + self._lambda_ntxent * ntx
                if log_aux:
                    self.log("train_ntxent" if self.training else "val_ntxent", ntx)

            return extra

        def _shared_step(self, batch, log_aux: bool = True):
            label = batch["label"].float()
            logit, video_emb, align_a, align_b = self._forward(batch)
            if self._loss_type == "focal":
                loss = focal_loss_with_logits(
                    logit,
                    label,
                    gamma=self._focal_gamma,
                    alpha=self._focal_alpha,
                    pos_weight=self._pos_weight,
                )
            else:
                loss = bce_with_logits(logit, label, self._pos_weight)
            loss = loss + self._contrastive_losses(
                video_emb, align_a, align_b, label, log_aux
            )
            proba = torch.sigmoid(logit)
            return loss, proba, label.int()

        def _bs(self, batch) -> int:
            return int(batch["label"].size(0))

        def training_step(self, batch, batch_idx):
            loss, _, _ = self._shared_step(batch)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=self._bs(batch))
            return loss

        def validation_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            log_val_metrics(self, loss, proba, label, self._bs(batch))

        def test_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            self.metrics.test_ap(proba, label)
            self.log("test_ap", self.metrics.test_ap, on_epoch=True)
            proba_2d = torch.cat([1 - proba, proba], dim=1)
            self.metrics.test_f1_macro(proba_2d, label.squeeze(1))
            self.log("test_f1_macro", self.metrics.test_f1_macro, on_epoch=True)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            logit, _, _, _ = self._forward(batch)
            proba = torch.sigmoid(logit)
            return {"video_ids": batch["video_id"], "proba": proba.squeeze(-1)}

        def configure_optimizers(self):
            return configure_adamw_scheduler(
                self.parameters(), self._lr, self._weight_decay, monitor="val_f1_macro"
            )

    return LitMultimodalHeteroFull()
