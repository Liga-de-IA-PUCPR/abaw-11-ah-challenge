import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision, MulticlassF1Score


class ClassificationMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_ap = BinaryAveragePrecision()
        self.val_f1_macro = MulticlassF1Score(num_classes=2, average="macro")
        self.test_acc = BinaryAccuracy()
        self.test_ap = BinaryAveragePrecision()
        self.test_f1_macro = MulticlassF1Score(num_classes=2, average="macro")

    def log_train(self, pl_module, loss, preds, target):
        pl_module.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc(preds, target)
        pl_module.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

    def log_val(self, pl_module, loss, preds, target):
        pl_module.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(preds, target)
        pl_module.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_ap(preds, target)
        pl_module.log("val_ap", self.val_ap, on_step=False, on_epoch=True, prog_bar=True)
        preds_2d = torch.cat([1 - preds, preds], dim=1)
        self.val_f1_macro(preds_2d, target.squeeze(1))
        pl_module.log("val_f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)

    def log_test(self, pl_module, loss, preds, target):
        pl_module.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_acc(preds, target)
        pl_module.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_ap(preds, target)
        pl_module.log("test_ap", self.test_ap, on_step=False, on_epoch=True)
        preds_2d = torch.cat([1 - preds, preds], dim=1)
        self.test_f1_macro(preds_2d, target.squeeze(1))
        pl_module.log("test_f1_macro", self.test_f1_macro, on_step=False, on_epoch=True)
