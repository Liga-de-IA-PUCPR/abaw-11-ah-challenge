import lightning as L
from torch import nn, optim

from evaluation.metrics import ClassificationMetrics


class BidirectionalCrossAttentionFusionModel(nn.Module):
    def __init__(self, dim_a, dim_b, common_dim=512, num_heads=4, num_classes=1):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, common_dim)
        self.proj_b = nn.Linear(dim_b, common_dim)
        self.cross_attn_a = nn.MultiheadAttention(
            embed_dim=common_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attn_b = nn.MultiheadAttention(
            embed_dim=common_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(common_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, num_classes),
        )

    def forward(self, feat_a, feat_b):
        # feat_a: (B, T1, D1), feat_b: (B, T2, D2)
        feat_a_proj = self.proj_a(feat_a)
        feat_b_proj = self.proj_b(feat_b)
        attn_out_a, _ = self.cross_attn_a(feat_a_proj, feat_b_proj, feat_b_proj)
        attn_out_b, _ = self.cross_attn_b(feat_b_proj, feat_a_proj, feat_a_proj)

        fused_a = self.norm(feat_a_proj + attn_out_a)
        fused_b = self.norm(feat_b_proj + attn_out_b)

        fused = self.norm(fused_a + fused_b)

        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)
        return self.classifier(pooled)


class BidirectionalCrossAttention(L.LightningModule):
    def __init__(self, cross_attention, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["cross_attention"])
        self.model = cross_attention
        self.metrics = ClassificationMetrics()

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.metrics.log_train(self, loss, preds, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.metrics.log_val(self, loss, preds, labels)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.metrics.log_test(self, loss, preds, labels)

    def _shared_step(self, batch):
        feat_a = batch["audio_emb"].unsqueeze(1)  # [B, 768] → [B, 1, 768]
        feat_b = batch["text_emb"].unsqueeze(1)  # [B, 384] → [B, 1, 384]
        label = batch["label"].float().unsqueeze(1)  # [B] → [B, 1]
        x_hat = self.model(feat_a, feat_b)
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, label)
        preds = nn.functional.sigmoid(x_hat)
        return loss, preds, label.int()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
