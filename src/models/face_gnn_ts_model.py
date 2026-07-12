"""Modelo isolado: apenas Face Mesh + GCN temporal (GNN4TS-style).

Treina separado do stack multimodal para validar a modelagem facial antes de
concatenar ao áudio/texto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.logger import get_logger
from src.models.face_gcn_ts import TemporalMode, _build_face_gcn_ts

if TYPE_CHECKING:
    import torch

log = get_logger("models.face_gnn_ts")


def _tab_mean_pool(
    tab_seq: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Média das features tabulares ao longo das janelas válidas."""
    import torch

    pooled: list[torch.Tensor] = []
    for i in range(tab_seq.size(0)):
        t = int(lengths[i].item())
        if t > 0:
            pooled.append(tab_seq[i, :t].mean(dim=0))
        else:
            pooled.append(tab_seq.new_zeros(tab_seq.size(-1)))
    return torch.stack(pooled, dim=0)


def _build_face_gnn_ts_module(
    dim_tab: int,
    use_tabular: bool,
    tab_hidden: int,
    dropout: float,
    *,
    face_top_k: int,
    face_spatial_hidden: int,
    face_spatial_out: int,
    face_temporal_hidden: int,
    face_temporal_out: int,
    temporal_mode: TemporalMode,
    use_velocity: bool,
):
    import torch
    from torch import nn

    face_encoder = _build_face_gcn_ts(
        spatial_hidden=face_spatial_hidden,
        spatial_out=face_spatial_out,
        temporal_hidden=face_temporal_hidden,
        temporal_out=face_temporal_out,
        top_k=face_top_k,
        dropout=dropout,
        temporal_mode=temporal_mode,
        use_velocity=use_velocity,
    )

    class _FaceGnnTsModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.face_encoder = face_encoder
            self.use_tabular = use_tabular
            self.dropout = nn.Dropout(dropout)
            readout_dim = face_encoder.out_dim
            if use_tabular:
                self.tab_proj = nn.Sequential(
                    nn.Linear(dim_tab, tab_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                readout_dim += tab_hidden
            self.readout_dim = readout_dim
            self.classifier = nn.Sequential(
                nn.Linear(readout_dim, readout_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(readout_dim, 1),
            )

        def forward(
            self,
            face_seq: torch.Tensor,
            lengths: torch.Tensor,
            tab_seq: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            parts = [self.face_encoder(face_seq, lengths)]
            if self.use_tabular and tab_seq is not None:
                tab_pool = _tab_mean_pool(tab_seq, lengths)
                parts.append(self.tab_proj(tab_pool))
            video_emb = self.dropout(torch.cat(parts, dim=-1))
            logit = self.classifier(video_emb)
            return logit, video_emb

    return _FaceGnnTsModule()


def build_face_lit_module(
    fusion,
    lr: float,
    weight_decay: float,
    pos_weight: float | None = None,
    contrastive_cfg: dict[str, Any] | None = None,
    loss_cfg: dict[str, Any] | None = None,
):
    """LightningModule face-only: BCE/focal + SupCon opcional no embedding de vídeo."""
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

    supcon_fn = None
    if use_contrastive and lambda_supcon > 0:
        from gnn_modalblocks.contrastive.registry import LOSSES

        temperature = float(ccfg.get("temperature", 0.07))
        supcon_fn = LOSSES.build("supervised_contrastive", temperature=temperature)

    class LitFaceGnnTs(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = fusion
            self.metrics = build_classification_metrics()
            self._lr = lr
            self._weight_decay = weight_decay
            self._pos_weight = pos_weight
            self._use_contrastive = use_contrastive
            self._lambda_supcon = lambda_supcon
            self.supcon_fn = supcon_fn
            self._loss_type = loss_type
            self._focal_gamma = focal_gamma
            self._focal_alpha = focal_alpha

        def _forward(self, batch):
            return self.model(
                batch["face_seq"],
                batch["lengths"],
                tab_seq=batch.get("tab_seq"),
            )

        def _shared_step(self, batch, log_aux: bool = True):
            label = batch["label"].float()
            logit, video_emb = self._forward(batch)
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

            if (
                self._use_contrastive
                and self.supcon_fn is not None
                and self._lambda_supcon > 0
            ):
                supcon = self.supcon_fn(video_emb, label.squeeze(-1).long())
                loss = loss + self._lambda_supcon * supcon
                if log_aux:
                    self.log("train_supcon" if self.training else "val_supcon", supcon)

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
            loss, proba, label = self._shared_step(batch, log_aux=False)
            self.metrics.test_ap(proba, label)
            self.log("test_ap", self.metrics.test_ap, on_epoch=True)
            proba_2d = torch.cat([1 - proba, proba], dim=1)
            self.metrics.test_f1_macro(proba_2d, label.squeeze(1))
            self.log("test_f1_macro", self.metrics.test_f1_macro, on_epoch=True)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            logit, _ = self._forward(batch)
            proba = torch.sigmoid(logit)
            return {"video_ids": batch["video_id"], "proba": proba.squeeze(-1)}

        def configure_optimizers(self):
            return configure_adamw_scheduler(
                self.parameters(), self._lr, self._weight_decay, monitor="val_f1_macro"
            )

    return LitFaceGnnTs()


class FaceGnnTsFusion:
    """Fachada registrável — classificador só com Face Mesh GCN temporal."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.dim_tab = int(cfg.get("dim_tab", 32))
        self.use_tabular = bool(cfg.get("use_tabular", True))
        self.tab_hidden = int(cfg.get("tab_hidden", 32))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))
        self.pos_weight = cfg.get("pos_weight", "auto")
        self._resolved_pos_weight: float | None = None

        face = cfg.get("face", {}) or {}
        self.face_top_k = int(face.get("top_k", 8))
        self.face_spatial_hidden = int(face.get("spatial_hidden", 64))
        self.face_spatial_out = int(face.get("spatial_out", 128))
        self.face_temporal_hidden = int(face.get("temporal_hidden", 64))
        self.face_temporal_out = int(face.get("temporal_out", 128))
        self.temporal_mode = str(face.get("temporal_mode", "gnn4ts"))
        self.use_velocity = bool(face.get("use_velocity", True))

        c = cfg.get("contrastive", {}) or {}
        self.contrastive = {
            "enabled": bool(c.get("enabled", False)),
            "lambda_supcon": float(c.get("lambda_supcon", 0.1)),
            "temperature": float(c.get("temperature", 0.07)),
        }
        loss = cfg.get("loss", {}) or {}
        self.loss = {
            "type": str(loss.get("type", "bce")),
            "gamma": float(loss.get("gamma", 2.0)),
            "alpha": loss.get("alpha", 0.25),
        }

    @classmethod
    def from_config(cls, config: Any) -> FaceGnnTsFusion:
        return cls(cfg=config)

    def build_module(self):
        return _build_face_gnn_ts_module(
            dim_tab=self.dim_tab,
            use_tabular=self.use_tabular,
            tab_hidden=self.tab_hidden,
            dropout=self.dropout,
            face_top_k=self.face_top_k,
            face_spatial_hidden=self.face_spatial_hidden,
            face_spatial_out=self.face_spatial_out,
            face_temporal_hidden=self.face_temporal_hidden,
            face_temporal_out=self.face_temporal_out,
            temporal_mode=self.temporal_mode,  # type: ignore[arg-type]
            use_velocity=self.use_velocity,
        )

    def build_lightning_module(self):
        return build_face_lit_module(
            fusion=self.build_module(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            pos_weight=self._resolved_pos_weight,
            contrastive_cfg=self.contrastive,
            loss_cfg=self.loss,
        )
