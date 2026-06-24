import lightning as L
import torch
from torch import nn, optim
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, dim_a, dim_b, common_dim=512, num_heads=4, num_classes=1):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, common_dim)
        self.proj_b = nn.Linear(dim_b, common_dim)
        self.cross_attn = nn.MultiheadAttention(
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
        q = self.proj_a(feat_a)
        kv = self.proj_b(feat_b)
        attn_out, _ = self.cross_attn(q, kv, kv)
        fused = self.norm(q + attn_out)
        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)
        return self.classifier(pooled)


class CrossAttention(L.LightningModule):
    def __init__(self, cross_attention, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["cross_attention"])
        self.model = cross_attention
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_auc = BinaryAUROC()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_auc = BinaryAUROC()

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_auc(preds, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_auc(preds, labels)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True)

    def _shared_step(self, batch):
        feat_a = batch["audio_emb"].unsqueeze(1)  # [B, 768] → [B, 1, 768]
        feat_b = batch["text_emb"].unsqueeze(1)  # [B, 384] → [B, 1, 384]
        label = batch["label"].float().unsqueeze(1)  # [B] → [B, 1]
        x_hat = self.model(feat_a, feat_b)
        loss = nn.functional.binary_cross_entropy_with_logits(x_hat, label)
        preds = torch.sigmoid(x_hat)
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
