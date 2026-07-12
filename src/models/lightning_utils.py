"""Utilitários compartilhados para LightningModules de classificação binária."""

from __future__ import annotations

from typing import Any

from src.logger import get_logger

log = get_logger("models.lightning_utils")


def compute_pos_weight_from_loader(loader) -> float:
    """Razão neg/pos a nível de vídeo no split de treino."""
    labels = loader.dataset.video_labels
    values = list(labels.values())
    n_pos = sum(1 for v in values if int(v) == 1)
    n_neg = len(values) - n_pos
    if n_pos == 0:
        log.warning("Nenhum positivo no treino; pos_weight=1.0")
        return 1.0
    ratio = n_neg / n_pos
    log.info(f"pos_weight auto: neg/pos = {n_neg}/{n_pos} = {ratio:.3f}")
    return float(ratio)


def resolve_pos_weight(setting: Any, train_loader) -> float | None:
    """Resolve ``model.pos_weight``: ``auto`` | float | ``null``/``none``."""
    if setting is None or setting in ("none", "null", False):
        return None
    if setting == "auto":
        return compute_pos_weight_from_loader(train_loader)
    return float(setting)


def bce_with_logits(logit, label, pos_weight: float | None):
    import torch
    from torch import nn

    pw = None
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=logit.device, dtype=logit.dtype)
    return nn.functional.binary_cross_entropy_with_logits(logit, label, pos_weight=pw)


def focal_loss_with_logits(
    logit,
    label,
    gamma: float = 2.0,
    alpha: float | None = 0.25,
    pos_weight: float | None = None,
):
    """Focal loss binária (Lin et al. 2017) — foca em exemplos difíceis.

    Reduz o peso dos exemplos fáceis (bem classificados) por ``(1 - p_t)^gamma``,
    concentrando o gradiente nos *hard positives/negatives*. ``alpha`` balanceia
    classes (peso da classe positiva); ``pos_weight`` é aplicado de forma
    multiplicativa como no BCE (compatível com ``pos_weight: auto``).
    """
    import torch
    from torch import nn

    pw = None
    if pos_weight is not None:
        pw = torch.tensor([pos_weight], device=logit.device, dtype=logit.dtype)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logit, label, pos_weight=pw, reduction="none"
    )
    p = torch.sigmoid(logit)
    p_t = p * label + (1 - p) * (1 - label)
    focal = (1 - p_t).clamp(min=1e-6) ** gamma * bce
    if alpha is not None:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        focal = alpha_t * focal
    return focal.mean()


def build_classification_metrics():
    from torch import nn
    from torchmetrics.classification import BinaryAveragePrecision, MulticlassF1Score

    class _ClassificationMetrics(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.val_ap = BinaryAveragePrecision()
            self.val_f1_macro = MulticlassF1Score(num_classes=2, average="macro")
            self.test_ap = BinaryAveragePrecision()
            self.test_f1_macro = MulticlassF1Score(num_classes=2, average="macro")

    return _ClassificationMetrics()


def configure_adamw_scheduler(
    params,
    lr: float,
    weight_decay: float,
    monitor: str = "val_loss",
):
    from torch import optim

    mode = "max" if "f1" in monitor.lower() else "min"
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=0.5, patience=10
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": monitor},
    }


def log_val_metrics(lit_module, loss, proba, label, batch_size: int) -> None:
    """Loga val_loss, val_ap e val_f1_macro (padrão cross_attention)."""
    import torch

    lit_module.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
    lit_module.metrics.val_ap(proba, label)
    lit_module.log("val_ap", lit_module.metrics.val_ap, on_epoch=True, prog_bar=True)
    proba_2d = torch.cat([1 - proba, proba], dim=1)
    lit_module.metrics.val_f1_macro(proba_2d, label.squeeze(1))
    lit_module.log(
        "val_f1_macro",
        lit_module.metrics.val_f1_macro,
        on_epoch=True,
        prog_bar=True,
    )
