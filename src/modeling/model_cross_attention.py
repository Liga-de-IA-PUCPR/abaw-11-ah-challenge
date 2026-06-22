import lightning as L
from torch import nn, optim


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, dim_a, dim_b, common_dim=512, num_heads=4, num_classes=2):
        super().__init__()
        # Project features to common dimension
        self.proj_a = nn.Linear(dim_a, common_dim)
        self.proj_b = nn.Linear(dim_b, common_dim)
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=common_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(common_dim)
        # Classifier head
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
    def __init__(self, cross_attention):
        super().__init__()
        self.model = cross_attention

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)

        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
