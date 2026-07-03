import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, optim

from evaluation.metrics import ClassificationMetrics
from modeling.local_attention.local_attention import LocalAttention


class LocalGlobalEncoder(nn.Module):
    """Local + global self-attention with learnable alpha fusion → common_dim"""

    def __init__(self, input_dim, common_dim, num_heads_local=8, window_size=512):
        super().__init__()
        assert common_dim % num_heads_local == 0
        self.num_heads = num_heads_local
        self.head_dim = common_dim // num_heads_local

        # local self-attention projections (LocalAttention has none internally)
        self.local_proj_q = nn.Linear(input_dim, common_dim)
        self.local_proj_k = nn.Linear(input_dim, common_dim)
        self.local_proj_v = nn.Linear(input_dim, common_dim)
        self.local_proj_o = nn.Linear(common_dim, common_dim)
        self.local_attn = LocalAttention(
            dim=self.head_dim,
            window_size=window_size,
            causal=False,
            look_backward=1,
            look_forward=1,
        )

        # global self-attention — project input first, then attend
        self.global_proj = nn.Linear(input_dim, common_dim)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=common_dim, num_heads=1, batch_first=True
        )

        # learnable fusion — scalar alpha, init at 0.5 via sigmoid(0)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(common_dim)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # local self-attention
        q = self.local_proj_q(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.local_proj_k(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.local_proj_v(x).reshape(B, T, H, D).transpose(1, 2)
        local_out = self.local_attn(q, k, v)  # (B, H, T, D)
        local_out = local_out.transpose(1, 2).reshape(B, T, H * D)
        local_out = self.local_proj_o(local_out)  # (B, T, common_dim)

        # global self-attention
        x_g = self.global_proj(x)  # (B, T, common_dim)
        global_out, _ = self.global_attn(x_g, x_g, x_g)  # (B, T, common_dim)

        # fuse
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * local_out + (1 - alpha) * global_out
        return self.norm(fused)  # (B, T, common_dim)


class BiCrossAttentionModel(nn.Module):
    def __init__(
        self, dim_a, dim_b, common_dim=512, num_heads=8, window_size=512, num_classes=1
    ):
        super().__init__()
        self.encoder_a = LocalGlobalEncoder(dim_a, common_dim, window_size=window_size)
        self.encoder_b = LocalGlobalEncoder(dim_b, common_dim, window_size=window_size)

        self.cross_attn_a = nn.MultiheadAttention(
            common_dim, num_heads, batch_first=True
        )
        self.cross_attn_b = nn.MultiheadAttention(
            common_dim, num_heads, batch_first=True
        )

        self.norm = nn.LayerNorm(common_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, num_classes),
        )

    def forward(self, feat_a, feat_b):
        a = self.encoder_a(feat_a)  # (B, T1, common_dim)
        b = self.encoder_b(feat_b)  # (B, T2, common_dim)

        attn_a, _ = self.cross_attn_a(a, b, b)
        attn_b, _ = self.cross_attn_b(b, a, a)

        fused_a = self.norm(a + attn_a)
        fused_b = self.norm(b + attn_b)

        pooled_a = self.pool(fused_a.transpose(1, 2)).squeeze(-1)
        pooled_b = self.pool(fused_b.transpose(1, 2)).squeeze(-1)

        return self.classifier(pooled_a + pooled_b)


class CrossAttention(L.LightningModule):
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
        feat_a = batch["audio_emb"]  # (B, T1, 768) — already sequential
        feat_b = batch["text_emb"]  # (B, T2, 384)
        label = batch["label"].float().unsqueeze(1)
        x_hat = self.model(feat_a, feat_b)
        loss = F.binary_cross_entropy_with_logits(x_hat, label)
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
