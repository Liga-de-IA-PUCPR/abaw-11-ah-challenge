"""Multimodal completo + Face Mesh GCN temporal (468 pts, GNN4TS-style)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.logger import get_logger
from src.models.multimodal_hetero_full import (
    MultimodalHeteroFullFusion,
    build_full_lit_module,
    _build_full_module,
)

if TYPE_CHECKING:
    import torch

log = get_logger("models.multimodal_hetero_face")


def _build_face_module(
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
    use_latent_gcn: bool,
    use_bilstm: bool,
    lstm_hidden: int,
    *,
    face_top_k: int = 8,
    face_spatial_hidden: int = 64,
    face_spatial_out: int = 128,
    face_temporal_hidden: int = 64,
    face_temporal_out: int = 128,
    face_temporal_mode: str = "chain",
    face_use_velocity: bool = False,
    face_landmark_stride: int = 1,
    face_max_windows: int | None = None,
):
    import torch
    from torch import nn

    from src.models.face_gcn_ts import _build_face_gcn_ts

    base = _build_full_module(
        dim_a=dim_a,
        dim_b=dim_b,
        dim_tab=dim_tab,
        common_dim=common_dim,
        hidden_channels=hidden_channels,
        heads=heads,
        out_channels=out_channels,
        gcn_out_dim=gcn_out_dim,
        top_k=top_k,
        fusion=fusion,
        dropout=dropout,
        use_latent_gcn=use_latent_gcn,
        use_bilstm=use_bilstm,
        lstm_hidden=lstm_hidden,
    )

    face_encoder = _build_face_gcn_ts(
        spatial_hidden=face_spatial_hidden,
        spatial_out=face_spatial_out,
        temporal_hidden=face_temporal_hidden,
        temporal_out=face_temporal_out,
        top_k=face_top_k,
        dropout=dropout,
        temporal_mode=face_temporal_mode,  # type: ignore[arg-type]
        use_velocity=face_use_velocity,
        landmark_stride=face_landmark_stride,
        max_windows=face_max_windows,
    )

    class _MultimodalHeteroFaceModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.face_encoder = face_encoder
            self.dropout = nn.Dropout(dropout)
            readout_dim = base.readout_dim + face_encoder.out_dim
            self.readout_dim = readout_dim
            self.classifier = nn.Sequential(
                nn.Linear(readout_dim, readout_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(readout_dim, 1),
            )

        def forward(
            self,
            feat_a: torch.Tensor,
            feat_b: torch.Tensor,
            tab_seq: torch.Tensor,
            lengths: torch.Tensor,
            face_seq: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            fused, proj_a, proj_b = self.base._fuse_windows(feat_a, feat_b)
            readout_parts = [self.base._gat_video_z(feat_a, feat_b, fused, tab_seq, lengths)]

            if self.base.use_latent_gcn and self.base.latent_gcn is not None:
                readout_parts.append(self.base._latent_gcn_pool(fused, lengths))
            if self.base.use_bilstm and self.base.temporal is not None:
                readout_parts.append(self.base._bilstm_pool(fused, lengths))
            if face_seq is not None and face_seq.numel() > 0:
                readout_parts.append(self.face_encoder(face_seq, lengths))

            video_emb = self.dropout(torch.cat(readout_parts, dim=-1))
            logit = self.classifier(video_emb)
            align_a, align_b = self.base._collect_align_pairs(proj_a, proj_b, lengths)
            return logit, video_emb, align_a, align_b

    return _MultimodalHeteroFaceModule()


class MultimodalHeteroFaceFusion(MultimodalHeteroFullFusion):
    """Stack multimodal_hetero_full + encoder Face Mesh GCN temporal."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        face = cfg.get("face", {}) or {}
        self.use_face = bool(face.get("enabled", True))
        self.face_top_k = int(face.get("top_k", 8))
        self.face_spatial_hidden = int(face.get("spatial_hidden", 64))
        self.face_spatial_out = int(face.get("spatial_out", 128))
        self.face_temporal_hidden = int(face.get("temporal_hidden", 64))
        self.face_temporal_out = int(face.get("temporal_out", 128))
        self.face_temporal_mode = str(face.get("temporal_mode", "chain"))
        self.face_use_velocity = bool(face.get("use_velocity", False))
        self.face_landmark_stride = int(face.get("landmark_stride", 1))
        mw = face.get("max_windows")
        self.face_max_windows = None if mw in (None, "null", "none", 0) else int(mw)

    def build_module(self):
        if not self.use_face:
            return super().build_module()
        module = _build_face_module(
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
            face_top_k=self.face_top_k,
            face_spatial_hidden=self.face_spatial_hidden,
            face_spatial_out=self.face_spatial_out,
            face_temporal_hidden=self.face_temporal_hidden,
            face_temporal_out=self.face_temporal_out,
            face_temporal_mode=self.face_temporal_mode,
            face_use_velocity=self.face_use_velocity,
            face_landmark_stride=self.face_landmark_stride,
            face_max_windows=self.face_max_windows,
        )
        from src.models.hetero_gnn import load_gae_projections

        gae_path = self._gae_init_path or self.gae_init
        if gae_path:
            load_gae_projections(module.base, str(gae_path))
        return module

    def build_lightning_module(self):
        lit = build_full_lit_module(
            fusion=self.build_module(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            pos_weight=self._resolved_pos_weight,
            contrastive_cfg=self.contrastive,
            loss_cfg=self.loss,
        )

        def _forward_with_face(batch):
            return lit.model(
                batch["audio_seq"],
                batch["text_seq"],
                batch["tab_seq"],
                batch["lengths"],
                face_seq=batch.get("face_seq"),
                key_padding_mask=batch.get("key_padding_mask"),
            )

        lit._forward = _forward_with_face  # noqa: SLF001
        return lit
