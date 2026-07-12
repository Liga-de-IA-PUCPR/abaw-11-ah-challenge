"""GNN baseline — mesma interface da cross-attention, blocos da gnn-modalblocks.

Pipeline por vídeo (espelha a cross-attention de ``luiz``):

  1. ``MultimodalBlock`` (attention) funde áudio+texto **por janela`` → nó temporal.
  2. ``LatentCorrelationGCN`` propaga no grafo em cadeia (janelas consecutivas).
  3. Pooling temporal mascarado → MLP → logit de vídeo.

Usa o **mesmo batch** que ``cross_attention`` (``collate_sequences``) — sem
``HeteroData`` no dataloader (evita o ``NotImplementedError`` do Lightning).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.logger import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger("models.gnn_baseline")


def _build_fusion_module(
    dim_a: int,
    dim_b: int,
    dim_tab: int,
    common_dim: int,
    gcn_out_dim: int,
    top_k: int,
    dropout: float,
    fusion: str,
    use_tabular: bool = True,
):
    import torch
    from torch import nn

    from gnn_modalblocks import ENCODERS, MultimodalBlock

    class _GnnBaselineModule(nn.Module):
        """Fusão multimodal por janela + GCN temporal + readout mascarado."""

        def __init__(self) -> None:
            super().__init__()
            self.proj_a = nn.Linear(dim_a, common_dim)
            self.proj_b = nn.Linear(dim_b, common_dim)
            self.multimodal = MultimodalBlock(
                modality_dims={"audio": common_dim, "text": common_dim},
                fusion=fusion,
                out_dim=common_dim,
            )
            self.gcn = ENCODERS.build(
                "latent_gcn",
                in_channels=common_dim,
                out_channels=gcn_out_dim,
                num_nodes=64,
                top_k=top_k,
            )
            self.dropout = nn.Dropout(dropout)
            tab_dim = min(common_dim // 4, max(dim_tab, 1)) if use_tabular else 0
            self.use_tabular = use_tabular and dim_tab > 0
            self.tab_proj = (
                nn.Linear(dim_tab, tab_dim) if self.use_tabular else None
            )
            clf_in = gcn_out_dim + (tab_dim if self.use_tabular else 0)
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, common_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(common_dim, 1),
            )

        @staticmethod
        def _masked_mean(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
            if key_padding_mask is None:
                return x.mean(dim=1)
            valid = (~key_padding_mask).unsqueeze(-1).float()
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        def forward(
            self,
            feat_a: torch.Tensor,
            feat_b: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
            tab_seq: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """``(B, T, D)`` → ``(B, 1)`` logits."""
            a = self.proj_a(feat_a)
            b = self.proj_b(feat_b)
            batch_size, t_max, channels = a.shape

            fused = self.multimodal(
                {
                    "audio": a.reshape(batch_size * t_max, channels),
                    "text": b.reshape(batch_size * t_max, channels),
                }
            ).view(batch_size, t_max, -1)

            pooled: list[torch.Tensor] = []
            for i in range(batch_size):
                if key_padding_mask is not None:
                    valid = ~key_padding_mask[i]
                    nodes = fused[i, valid]
                else:
                    nodes = fused[i]
                if nodes.size(0) == 0:
                    pooled.append(torch.zeros(gcn_out_dim, device=fused.device))
                    continue
                h = self.gcn(nodes)
                pooled.append(h.mean(dim=0))

            video_emb = self.dropout(torch.stack(pooled))
            if self.use_tabular and tab_seq is not None and self.tab_proj is not None:
                tab_pooled = self.tab_proj(self._masked_mean(tab_seq, key_padding_mask))
                video_emb = torch.cat([video_emb, tab_pooled], dim=-1)
            return self.classifier(video_emb)

    return _GnnBaselineModule()


class GnnBaselineFusion:
    """Fachada registrável (family=lightning) para o GNN baseline."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.dim_a = int(cfg.get("dim_a", 768))
        self.dim_b = int(cfg.get("dim_b", 768))
        self.dim_tab = int(cfg.get("dim_tab", 32))
        self.common_dim = int(cfg.get("common_dim", 512))
        self.gcn_out_dim = int(cfg.get("gcn_out_dim", 128))
        self.top_k = int(cfg.get("top_k", 8))
        self.fusion = str(cfg.get("fusion", "attention"))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))
        self.pos_weight = cfg.get("pos_weight", "auto")
        self.use_tabular = bool(cfg.get("use_tabular", True))
        self._resolved_pos_weight: float | None = None

    @classmethod
    def from_config(cls, config: Any) -> GnnBaselineFusion:
        return cls(cfg=config)

    def build_module(self):
        return _build_fusion_module(
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            dim_tab=self.dim_tab,
            common_dim=self.common_dim,
            gcn_out_dim=self.gcn_out_dim,
            top_k=self.top_k,
            dropout=self.dropout,
            fusion=self.fusion,
            use_tabular=self.use_tabular,
        )

    def build_lightning_module(self):
        from src.models.lightning_seq import build_sequence_lit_module

        return build_sequence_lit_module(
            fusion=self.build_module(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            pos_weight=self._resolved_pos_weight,
            use_tabular=self.use_tabular,
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
