import lightning as L
from torch import nn, optim


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

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def _shared_step(self, batch):
        feat_a = batch["audio_emb"].unsqueeze(1)  # [B, 768] → [B, 1, 768]
        feat_b = batch["text_emb"].unsqueeze(1)  # [B, 384] → [B, 1, 384]
        label = batch["label"].float().unsqueeze(1)  # [B] → [B, 1]
        x_hat = self.model(feat_a, feat_b)
        return nn.functional.binary_cross_entropy_with_logits(x_hat, label)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)
