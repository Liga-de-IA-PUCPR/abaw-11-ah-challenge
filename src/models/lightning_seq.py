"""LightningModule compartilhado para modelos de sequência ``(B, T, D)``."""

from __future__ import annotations

from src.models.lightning_utils import (
    bce_with_logits,
    build_classification_metrics,
    configure_adamw_scheduler,
    log_val_metrics,
)


def build_sequence_lit_module(
    fusion,
    lr: float,
    weight_decay: float,
    pos_weight: float | None = None,
    use_tabular: bool = False,
):
    """``LitSequenceModel`` para fusões com ``forward(a, b, key_padding_mask[, tab_seq])``."""
    import lightning as L
    import torch

    class LitSequenceModel(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = fusion
            self.metrics = build_classification_metrics()
            self._lr = lr
            self._weight_decay = weight_decay
            self._pos_weight = pos_weight
            self._use_tabular = use_tabular

        def _forward(self, batch):
            feat_a = batch["audio_seq"]
            feat_b = batch["text_seq"]
            mask = batch["key_padding_mask"]
            if self._use_tabular:
                return self.model(
                    feat_a,
                    feat_b,
                    key_padding_mask=mask,
                    tab_seq=batch["tab_seq"],
                )
            return self.model(feat_a, feat_b, key_padding_mask=mask)

        def _shared_step(self, batch):
            label = batch["label"].float()
            logit = self._forward(batch)
            loss = bce_with_logits(logit, label, self._pos_weight)
            proba = torch.sigmoid(logit)
            return loss, proba, label.int()

        def _batch_size(self, batch) -> int:
            return int(batch["label"].size(0))

        def training_step(self, batch, batch_idx):
            loss, _, _ = self._shared_step(batch)
            bs = self._batch_size(batch)
            self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=bs)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            log_val_metrics(self, loss, proba, label, self._batch_size(batch))

        def test_step(self, batch, batch_idx):
            loss, proba, label = self._shared_step(batch)
            bs = self._batch_size(batch)
            self.log("test_loss", loss, on_epoch=True, batch_size=bs)
            self.metrics.test_ap(proba, label)
            self.log("test_ap", self.metrics.test_ap, on_epoch=True)
            proba_2d = torch.cat([1 - proba, proba], dim=1)
            self.metrics.test_f1_macro(proba_2d, label.squeeze(1))
            self.log("test_f1_macro", self.metrics.test_f1_macro, on_epoch=True)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            _, proba, _ = self._shared_step(batch)
            return {"video_ids": batch["video_id"], "proba": proba.squeeze(-1)}

        def configure_optimizers(self):
            return configure_adamw_scheduler(
                self.parameters(), self._lr, self._weight_decay
            )

    return LitSequenceModel()
