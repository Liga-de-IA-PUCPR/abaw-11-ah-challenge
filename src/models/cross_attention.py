"""CrossAttentionFusion + LitCrossAttention — modelo de competição (família lightning).

UPGRADE do ``model_cross_attention.py`` da branch ``matheus`` (que recebia UMA
janela por vídeo, T=1) para a **sequência de janelas** do vídeo (T = nº de janelas
> 1, README §6.2), fazendo modelagem temporal real.

Arquitetura (sobre tensores ``(B, T, D)``):
  proj_a/proj_b -> common_dim
  MultiheadAttention(q=áudio, kv=texto, key_padding_mask=máscara de padding)
  residual + LayerNorm
  POOLING TEMPORAL MASCARADO sobre T (média ignorando janelas de padding)
  MLP -> 1 logit por vídeo (BCEWithLogits).

⚠️ **Imports lazy:** ``torch``/``lightning``/``torchmetrics`` são importados DENTRO
das funções/métodos. O módulo só é importado pelo registry quando
``model=cross_attention`` — mantendo o caminho RandomForest livre dessas deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.logger import get_logger

if TYPE_CHECKING:  # só p/ type-checkers; não executa em runtime
    import torch

log = get_logger("models.cross_attention")


# =============================================================================
# Construtor lazy do nn.Module de fusão
# =============================================================================
def _build_fusion_module(
    dim_a: int,
    dim_b: int,
    common_dim: int,
    num_heads: int,
    dropout: float,
):
    """Constrói o ``nn.Module`` de cross-attention (importa torch *lazy*).

    Definido como factory para que ``import torch`` só ocorra quando a
    cross-attention é efetivamente instanciada (não no import do módulo).
    """
    from torch import nn

    class _CrossAttentionFusionModule(nn.Module):
        """Fusão cross-attention sobre sequência de janelas ``(B, T, D)`` com masking."""

        def __init__(self) -> None:
            super().__init__()
            self.proj_a = nn.Linear(dim_a, common_dim)  # áudio -> common
            self.proj_b = nn.Linear(dim_b, common_dim)  # texto -> common
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=common_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(common_dim)
            self.classifier = nn.Sequential(
                nn.Linear(common_dim, common_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(common_dim, 1),  # 1 logit por vídeo
            )

        def forward(
            self,
            feat_a: torch.Tensor,  # (B, T, dim_a) — áudio por janela
            feat_b: torch.Tensor,  # (B, T, dim_b) — texto por janela
            key_padding_mask: torch.Tensor | None = None,  # (B, T) True = padding
        ) -> torch.Tensor:
            """Devolve ``(B, 1)`` logits a nível de vídeo.

            ``key_padding_mask[b, t] = True`` indica janela de padding (variável T).
            """
            q = self.proj_a(feat_a)  # (B, T, C)  query = áudio
            kv = self.proj_b(feat_b)  # (B, T, C)  key/value = texto
            attn_out, _ = self.cross_attn(
                q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
            )
            fused = self.norm(q + attn_out)  # residual + LN -> (B, T, C)
            pooled = self._masked_mean(fused, key_padding_mask)  # (B, C)
            return self.classifier(pooled)  # (B, 1)

        @staticmethod
        def _masked_mean(
            x: torch.Tensor, key_padding_mask: torch.Tensor | None
        ) -> torch.Tensor:
            """Pooling temporal mascarado sobre T (média ignorando padding)."""
            if key_padding_mask is None:
                return x.mean(dim=1)
            valid = (~key_padding_mask).unsqueeze(-1).float()  # (B, T, 1)
            summed = (x * valid).sum(dim=1)  # (B, C)
            count = valid.sum(dim=1).clamp_min(1.0)  # (B, 1)
            return summed / count

    return _CrossAttentionFusionModule()


# =============================================================================
# Fachada registrável (sklearn-like .from_config) — sem herdar nn.Module
# =============================================================================
class CrossAttentionFusion:
    """Fachada da cross-attention para o registry (family="lightning").

    Guarda os hiperparâmetros e constrói o ``nn.Module`` *lazy* via ``build_module``.
    O ``LightningTrainer`` chama ``build_lightning_module`` para obter o
    ``LightningModule`` treinável (``LitCrossAttention``).
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.dim_a = int(cfg.get("dim_a", 768))  # áudio (wav2vec2/hubert)
        self.dim_b = int(cfg.get("dim_b", 768))  # texto (roberta-emotion)
        self.common_dim = int(cfg.get("common_dim", 512))
        self.num_heads = int(cfg.get("num_heads", 4))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))

    @classmethod
    def from_config(cls, config: Any) -> CrossAttentionFusion:
        """Lê ``configs/model/cross_attention.yaml`` (README §7, grupo ``model``)."""
        return cls(cfg=config)

    def build_module(self):
        """Instancia o ``nn.Module`` de fusão (importa torch *lazy*)."""
        return _build_fusion_module(
            dim_a=self.dim_a,
            dim_b=self.dim_b,
            common_dim=self.common_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

    def build_lightning_module(self):
        """Monta o ``LitCrossAttention`` pronto para ``L.Trainer.fit`` (lazy)."""
        return _build_lit_module(
            fusion=self.build_module(), lr=self.lr, weight_decay=self.weight_decay
        )

    def load_from_checkpoint(self, ckpt_path, map_location=None):
        """Reconstrói o ``nn.Module`` de fusão e carrega os pesos de um ``.ckpt`` Lightning.

        Helper de inferência (FASE 5 ``_predict_neural``). Como o ``LitCrossAttention`` é
        uma classe **closure** (definida dentro de ``_build_lit_module``, sem
        ``save_hyperparameters``) e **não** é exportada, ``LitCrossAttention.load_from_checkpoint``
        não é alcançável; aqui reconstruímos o ``nn.Module`` a partir do ``cfg`` desta fachada
        e carregamos o ``state_dict`` do checkpoint. Mantém ``torch``/``lightning`` *lazy*.

        O ``LitCrossAttention`` guarda o módulo de fusão em ``self.model`` (ver
        ``_build_lit_module``), então as chaves do ``state_dict`` têm prefixo ``model.``.

        Args:
            ckpt_path: caminho do arquivo ``.ckpt`` salvo pelo ``ModelCheckpoint``.
            map_location: device de destino (``torch.device`` ou str) p/ ``torch.load``.

        Returns:
            ``nn.Module`` de fusão em modo ``eval()`` — callable como
            ``module(feat_a, feat_b, key_padding_mask=...)`` → ``(B, 1)`` logits.
        """
        import torch

        ckpt = torch.load(str(ckpt_path), map_location=map_location)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        module = self.build_module()
        # O LightningModule expõe a fusão em ``self.model`` → remove o prefixo "model.".
        prefix = "model."
        fusion_state = {
            k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        # Fallback: se não houver prefixo (ckpt já só com a fusão), usa o state_dict inteiro.
        module.load_state_dict(fusion_state or state_dict)
        module.eval()
        return module


# =============================================================================
# LightningModule (construído lazy) — reusa evaluation/metrics.py do matheus
# =============================================================================
def _build_lit_module(fusion, lr: float, weight_decay: float):
    """Constrói o ``LitCrossAttention`` (importa lightning/torch/torchmetrics *lazy*).

    Reusa o ``ClassificationMetrics`` (torchmetrics: BinaryAveragePrecision +
    MulticlassF1Score macro) portado de ``src/evaluation/metrics.py`` do ``matheus``.
    """
    import lightning as L
    import torch
    from torch import nn, optim
    from torchmetrics.classification import BinaryAveragePrecision, MulticlassF1Score

    class _ClassificationMetrics(nn.Module):
        """Métricas torchmetrics do caminho Lightning (reuso do matheus)."""

        def __init__(self) -> None:
            super().__init__()
            self.val_ap = BinaryAveragePrecision()
            self.val_f1_macro = MulticlassF1Score(num_classes=2, average="macro")
            self.test_ap = BinaryAveragePrecision()
            self.test_f1_macro = MulticlassF1Score(num_classes=2, average="macro")

    class LitCrossAttention(L.LightningModule):
        """LightningModule da cross-attention temporal (1 logit por vídeo, BCE)."""

        def __init__(self) -> None:
            super().__init__()
            self.model = fusion
            self.metrics = _ClassificationMetrics()
            self._lr = lr
            self._weight_decay = weight_decay

        # ---- passos -------------------------------------------------------
        def _shared_step(self, batch):
            """Batch do ``VideoSequenceDataset``: tensores empilhados (B, T, D) + máscara."""
            feat_a = batch["audio_seq"]  # (B, T, dim_a)
            feat_b = batch["text_seq"]  # (B, T, dim_b)
            mask = batch["key_padding_mask"]  # (B, T) True = padding
            label = batch["label"].float()  # (B, 1) já vem do collate_sequences
            logit = self.model(feat_a, feat_b, key_padding_mask=mask)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, label)
            proba = torch.sigmoid(logit)  # (B, 1)
            return loss, proba, label.int()

        def training_step(self, batch, batch_idx):
            loss, _, _ = self._shared_step(batch)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.metrics.val_ap(proba, label)
            self.log("val_ap", self.metrics.val_ap, on_epoch=True, prog_bar=True)
            proba_2d = torch.cat([1 - proba, proba], dim=1)  # (B, 2)
            self.metrics.val_f1_macro(proba_2d, label.squeeze(1))
            self.log("val_f1_macro", self.metrics.val_f1_macro, on_epoch=True, prog_bar=True)

        def test_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            self.log("test_loss", loss, on_epoch=True)
            self.metrics.test_ap(proba, label)
            self.log("test_ap", self.metrics.test_ap, on_epoch=True)
            proba_2d = torch.cat([1 - proba, proba], dim=1)
            self.metrics.test_f1_macro(proba_2d, label.squeeze(1))
            self.log("test_f1_macro", self.metrics.test_f1_macro, on_epoch=True)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            """Devolve ``(video_ids, proba)`` para agregação/calibração no trainer."""
            _, proba, _ = self._shared_step(batch)
            return {"video_ids": batch["video_id"], "proba": proba.squeeze(1)}

        # ---- otimização ---------------------------------------------------
        def configure_optimizers(self):
            optimizer = optim.AdamW(
                self.parameters(), lr=self._lr, weight_decay=self._weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }

    return LitCrossAttention()
