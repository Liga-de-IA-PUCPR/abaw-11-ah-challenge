"""HeteroGAT + contrastive multi-task (BCE + SupervisedContrastive + triplet opcional)."""

from __future__ import annotations

from typing import Any

from src.logger import get_logger
from src.models.hetero_gnn import (
    HeteroGnnFusion,
    _build_gnn_module,
    build_hetero_lit_module,
    load_gae_projections,
)

log = get_logger("models.hetero_gnn_contrastive")


class HeteroGnnContrastiveFusion(HeteroGnnFusion):
    """HeteroGAT com loss composta: BCE + λ·SupCon (+ μ·triplet opcional)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        c = cfg.get("contrastive", {}) or {}
        self.contrastive = {
            "enabled": bool(c.get("enabled", True)),
            "lambda_supcon": float(c.get("lambda_supcon", 0.1)),
            "lambda_triplet": float(c.get("lambda_triplet", 0.0)),
            "temperature": float(c.get("temperature", 0.07)),
            "margin": float(c.get("margin", 0.2)),
            "miner": str(c.get("miner", "batch_hard")),
        }

    @classmethod
    def from_config(cls, config: Any) -> HeteroGnnContrastiveFusion:
        return cls(cfg=config)

    def build_module(self):
        use_contrastive = bool(self.contrastive.get("enabled", True))
        module = _build_gnn_module(
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            dim_tab=self.dim_tab,
            hidden_channels=self.hidden_channels,
            heads=self.heads,
            out_channels=self.out_channels,
            dropout=self.dropout,
            gat_num_layers=self.gat_num_layers,
            return_embedding=use_contrastive,
            use_tab_enhanced=self.use_tab_enhanced,
            tab_pool=self.tab_pool,
            common_dim=self.common_dim,
            use_ca_edge_weights=self.use_ca_edge_weights,
            ca_num_heads=self.ca_num_heads,
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
            contrastive_cfg=self.contrastive,
            loss_cfg=self.loss,
            trainer_cfg=trainer_cfg,
        )
