"""GNN heterogêneo (avançado) — HeteroGAT via gnn-modalblocks.

Mesmo batch que cross-attention; o grafo é montado **dentro** do ``forward``
(evita ``HeteroData`` no dataloader e o bug do Lightning).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.data.graph_builder import BAH_GRAPH_METADATA, build_video_hetero_graph
from src.logger import get_logger
from src.models.lightning_utils import (
    build_classification_metrics,
    configure_adamw_scheduler,
    log_val_metrics,
)

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
    gat_num_layers: int = 2,
    return_embedding: bool = False,
):
    import torch
    from torch import nn
    from torch_geometric.data import Batch

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
        num_layers=gat_num_layers,
    )

    class _BahVideoHeteroGNN(nn.Module):
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
            self._return_embedding = return_embedding

        def _encode_batch(
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
            _proba, z_dict = self.gat(batch_graph)
            video_z = z_dict.get("video")
            if video_z is None or video_z.size(0) == 0:
                return torch.zeros(
                    feat_a.size(0),
                    self.refine[0].in_features,
                    device=feat_a.device,
                )
            return video_z

        def forward(
            self,
            feat_a: torch.Tensor,
            feat_b: torch.Tensor,
            tab_seq: torch.Tensor,
            lengths: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            video_z = self._encode_batch(feat_a, feat_b, tab_seq, lengths)
            logit = self.refine(self.dropout(video_z))
            if self._return_embedding:
                return logit, video_z
            return logit

    return _BahVideoHeteroGNN()


def load_gae_projections(gnn_module, gae_ckpt_path: str) -> None:
    """Inicializa projeções do HeteroGAT a partir do encoder GAE pré-treinado."""
    import torch

    state = torch.load(str(gae_ckpt_path), map_location="cpu", weights_only=False)
    enc_state = state.get("encoder", state)
    gat_proj = gnn_module.gat.projections.state_dict()
    loaded = 0
    for key, val in enc_state.items():
        if not key.startswith("projections."):
            continue
        tgt = key.replace("projections.", "")
        if tgt in gat_proj and gat_proj[tgt].shape == val.shape:
            gat_proj[tgt] = val
            loaded += 1
    if loaded:
        gnn_module.gat.projections.load_state_dict(gat_proj, strict=False)
        log.info(f"GAE init: {loaded} tensores de projeção carregados de {gae_ckpt_path}")
    else:
        log.warning(f"GAE init: nenhum tensor compatível em {gae_ckpt_path}")


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
        self.gat_num_layers = int(cfg.get("gat_num_layers", 2))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))
        self.pos_weight = cfg.get("pos_weight", "auto")
        self.gae_init = cfg.get("gae_init", None)
        loss = cfg.get("loss", {}) or {}
        self.loss = {
            "type": str(loss.get("type", "bce")),
            "gamma": float(loss.get("gamma", 2.0)),
            "alpha": loss.get("alpha", 0.25),
            "label_smoothing": float(loss.get("label_smoothing", 0.0)),
        }
        self._resolved_pos_weight: float | None = None
        self._gae_init_path: str | None = None

    @classmethod
    def from_config(cls, config: Any) -> HeteroGnnFusion:
        return cls(cfg=config)

    def build_module(self):
        module = _build_gnn_module(
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            dim_tab=self.dim_tab,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            out_channels=self.out_channels,
            dropout=self.dropout,
            gat_num_layers=self.gat_num_layers,
        )
        gae_path = self._gae_init_path or self.gae_init
        if gae_path:
            load_gae_projections(module, str(gae_path))
        return module

    def build_lightning_module(self, trainer_cfg: dict[str, Any] | None = None):
        return build_hetero_lit_module(
            fusion=self.build_module(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            pos_weight=self._resolved_pos_weight,
            loss_cfg=self.loss,
            trainer_cfg=trainer_cfg,
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


def build_hetero_lit_module(
    fusion,
    lr: float,
    weight_decay: float,
    pos_weight: float | None = None,
    contrastive_cfg: dict[str, Any] | None = None,
    loss_cfg: dict[str, Any] | None = None,
    trainer_cfg: dict[str, Any] | None = None,
):
    """LightningModule para HeteroGAT (opcionalmente multi-task contrastive)."""
    import lightning as L
    import torch

    ccfg = contrastive_cfg or {}
    use_contrastive = bool(ccfg.get("enabled", False))
    lambda_supcon = float(ccfg.get("lambda_supcon", 0.1))
    lambda_triplet = float(ccfg.get("lambda_triplet", 0.0))
    temperature = float(ccfg.get("temperature", 0.07))
    margin = float(ccfg.get("margin", 0.2))
    miner_key = str(ccfg.get("miner", "batch_hard"))

    lcfg = loss_cfg or {}
    loss_type = str(lcfg.get("type", "bce")).lower()
    focal_gamma = float(lcfg.get("gamma", 2.0))
    focal_alpha = lcfg.get("alpha", 0.25)
    focal_alpha = None if focal_alpha in (None, "none", "null") else float(focal_alpha)
    label_smoothing = float(lcfg.get("label_smoothing", 0.0))

    tcfg = trainer_cfg or {}
    monitor = str(tcfg.get("monitor", "val_loss"))
    scheduler = str(tcfg.get("scheduler", "plateau"))
    max_epochs = int(tcfg.get("max_epochs", 200))
    warmup_epochs = int(tcfg.get("warmup_epochs", 5))
    min_lr = float(tcfg.get("min_lr", 1e-6))

    supcon_fn = None
    triplet_fn = None
    miner = None
    if use_contrastive:
        from gnn_modalblocks.contrastive.registry import LOSSES, MINERS

        supcon_fn = LOSSES.build("supervised_contrastive", temperature=temperature)
        if lambda_triplet > 0:
            triplet_fn = LOSSES.build("triplet_margin", margin=margin)
            miner = MINERS.build(miner_key)

    class LitHeteroGnn(L.LightningModule):
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
            self.supcon_fn = supcon_fn
            self.triplet_fn = triplet_fn
            self.miner = miner

        def _forward_logits(self, batch):
            out = self.model(
                batch["audio_seq"],
                batch["text_seq"],
                batch["tab_seq"],
                batch["lengths"],
                key_padding_mask=batch["key_padding_mask"],
            )
            if self._use_contrastive:
                logit, video_emb = out
                return logit, video_emb
            return out, None

        def _shared_step(self, batch, log_aux: bool = True):
            label = batch["label"].float()
            logit, video_emb = self._forward_logits(batch)

            from src.models.lightning_utils import classification_loss

            bce = classification_loss(
                logit,
                label,
                loss_type=loss_type,
                pos_weight=self._pos_weight,
                gamma=focal_gamma,
                alpha=focal_alpha,
                label_smoothing=label_smoothing,
            )
            loss = bce
            if (
                log_aux
                and self._use_contrastive
                and video_emb is not None
                and self.supcon_fn is not None
            ):
                supcon = self.supcon_fn(video_emb, label.squeeze(-1).long())
                loss = loss + self._lambda_supcon * supcon
                self.log("train_supcon" if self.training else "val_supcon", supcon)
                if (
                    self._lambda_triplet > 0
                    and self.triplet_fn is not None
                    and self.miner is not None
                ):
                    a, p, n = self.miner(video_emb, label.squeeze(-1).long())
                    trip = self.triplet_fn(a, p, n)
                    loss = loss + self._lambda_triplet * trip
                    self.log("train_triplet" if self.training else "val_triplet", trip)

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
            logit, _ = self._forward_logits(batch)
            proba = torch.sigmoid(logit)
            return {"video_ids": batch["video_id"], "proba": proba.squeeze(-1)}

        def configure_optimizers(self):
            return configure_adamw_scheduler(
                self.parameters(),
                self._lr,
                self._weight_decay,
                monitor=monitor,
                scheduler=scheduler,
                max_epochs=max_epochs,
                warmup_epochs=warmup_epochs,
                min_lr=min_lr,
            )

    return LitHeteroGnn()
