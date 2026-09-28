"""Fine-tune do RoBERTa-GoEmotions (Rodada 1 do plano — "texto como âncora").

Receita inspirada nas duas equipes top do BAH (ver
``references/visual_signal_lessons_from_top_teams.md``):

  - Backbone ``SamLowe/roberta-base-go_emotions`` (28 emoções), no lugar do
    ``cardiffnlp/twitter-roberta-base-emotion`` congelado usado hoje
    (``configs/text_embedder/roberta_emotion.yaml``).
  - Embeddings + as N primeiras camadas do encoder CONGELADAS (default N=4);
    o resto é fine-tuned com lr baixo (2e-5).
  - CLS pooling sobre a transcrição COMPLETA (não só a janela de 5s).
  - Cabeça principal: MLP pequeno -> 1 logit (A/H), BCE com label smoothing
    e balanceamento de classe (``pos_weight``).
  - R-Drop (Liang et al. 2021): dois forward passes com dropout independente
    + termo de consistência KL simétrico entre as duas distribuições.
  - Cabeça auxiliar (estilo AMF): projeta o vetor de hesitação/tabular (74-d,
    ver ``src/features/HESITATION.md``) para 1 logit, com sua própria BCE,
    somada à loss total com peso ``aux_loss_weight`` (default 0.3).

Como ler este arquivo:
  1. ``TextFinetuneNet`` — o modelo puro (``nn.Module``): recebe tokens,
     devolve ``(logit, aux_logit)``.
  2. ``LitTextFinetune`` — o ``LightningModule`` que treina ``TextFinetuneNet``
     (loss, otimizador, logging).
  3. ``TextFinetune`` — a fachada que o registry Hydra chama (lê o config,
     baixa o backbone da HuggingFace, monta os dois objetos acima).
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from src.logger import get_logger
from src.models.lightning_utils import (
    bce_with_logits,
    build_classification_metrics,
    configure_adamw_scheduler,
    log_val_metrics,
    smooth_binary_labels,
)

log = get_logger("models.text_finetune")

DEFAULT_MODEL_NAME = "SamLowe/roberta-base-go_emotions"


def freeze_backbone_layers(backbone: nn.Module, num_frozen_layers: int) -> None:
    """Congela ``backbone.embeddings`` + as ``num_frozen_layers`` primeiras camadas.

    Espera um backbone estilo ``RobertaModel`` (``.embeddings`` +
    ``.encoder.layer`` indexável). Pode ser chamado de novo com outro N.
    """
    for param in backbone.embeddings.parameters():
        param.requires_grad = False
    layers = backbone.encoder.layer
    n = max(0, min(int(num_frozen_layers), len(layers)))
    for layer in layers[:n]:
        for param in layer.parameters():
            param.requires_grad = False
    log.info(f"Backbone: embeddings + {n}/{len(layers)} camadas congeladas.")


def symmetric_kl_from_logits(logit1: torch.Tensor, logit2: torch.Tensor) -> torch.Tensor:
    """KL simétrico entre duas distribuições Bernoulli(logit), para R-Drop.

    Trata cada logit binário como uma distribuição de 2 classes
    ``[-logit, logit]`` e calcula ``(KL(p1‖p2) + KL(p2‖p1)) / 2``. É 0
    quando ``logit1 == logit2`` (ex.: dropout desligado em ``eval()``).
    """
    dist1 = torch.cat([-logit1, logit1], dim=-1)
    dist2 = torch.cat([-logit2, logit2], dim=-1)
    log_p1 = F.log_softmax(dist1, dim=-1)
    log_p2 = F.log_softmax(dist2, dim=-1)
    kl_12 = F.kl_div(log_p1, log_p2.exp(), reduction="batchmean")
    kl_21 = F.kl_div(log_p2, log_p1.exp(), reduction="batchmean")
    return (kl_12 + kl_21) / 2.0


class TextFinetuneNet(nn.Module):
    """CLS pooling -> cabeça principal (A/H) + cabeça auxiliar (hesitação).

    Args:
        backbone: modelo HuggingFace tipo RoBERTa já instanciado
            (``AutoModel.from_pretrained(...)`` em produção, ou um
            ``RobertaModel`` tiny/random-init em testes).
        hidden_size: dimensão do ``last_hidden_state`` do backbone.
        dropout: dropout da cabeça principal.
        head_hidden: largura da camada oculta da cabeça principal.
        dim_tab: dimensão do vetor tabular de hesitação (0 desliga a
            cabeça auxiliar).
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        head_hidden: int = 256,
        dim_tab: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.aux_head = nn.Linear(dim_tab, 1) if dim_tab > 0 else None

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        tab: torch.Tensor | None = None,  # (B, dim_tab)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Devolve ``(logit, aux_logit)``, ambos ``(B, 1)``.

        ``aux_logit`` é ``None`` se ``dim_tab=0`` ou ``tab`` não for passado.
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logit = self.head(cls)
        aux_logit = self.aux_head(tab) if self.aux_head is not None and tab is not None else None
        return logit, aux_logit


class LitTextFinetune(L.LightningModule):
    """Treina ``TextFinetuneNet`` com R-Drop + cabeça auxiliar de hesitação.

    Loss total = BCE(principal, label smoothing) + ``rdrop_alpha``·KL(R-Drop)
    + ``aux_loss_weight``·BCE(cabeça auxiliar).
    """

    def __init__(
        self,
        net: TextFinetuneNet,
        lr: float = 2e-5,
        weight_decay: float = 1e-2,
        pos_weight: float | None = None,
        label_smoothing: float = 0.1,
        rdrop_alpha: float = 4.0,
        aux_loss_weight: float = 0.3,
        scheduler: str = "cosine_warmup",
        warmup_epochs: int = 2,
    ) -> None:
        super().__init__()
        self.net = net
        self.metrics = build_classification_metrics()
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.rdrop_alpha = rdrop_alpha
        self.aux_loss_weight = aux_loss_weight
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs

    def _step(self, batch: dict, rdrop: bool) -> tuple:
        """``rdrop=True`` faz 2 forward passes (treino); ``False`` faz 1 (val/test/predict)."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        tab = batch.get("tab")
        label = batch["label"].float()  # (B, 1)
        target = smooth_binary_labels(label, self.label_smoothing)

        logit1, aux_logit = self.net(input_ids, attention_mask, tab=tab)
        loss1 = bce_with_logits(logit1, target, self.pos_weight)

        if rdrop:
            logit2, _ = self.net(input_ids, attention_mask, tab=tab)
            loss2 = bce_with_logits(logit2, target, self.pos_weight)
            main_loss = 0.5 * (loss1 + loss2)
            kl = symmetric_kl_from_logits(logit1, logit2)
        else:
            main_loss = loss1
            kl = torch.zeros((), device=logit1.device, dtype=logit1.dtype)

        loss = main_loss + self.rdrop_alpha * kl

        aux_loss = None
        if aux_logit is not None:
            aux_loss = F.binary_cross_entropy_with_logits(aux_logit, label)
            loss = loss + self.aux_loss_weight * aux_loss

        proba = torch.sigmoid(logit1)  # (B, 1)
        return loss, main_loss, kl, aux_loss, proba, label.int()

    def training_step(self, batch, batch_idx):
        loss, main_loss, kl, aux_loss, _, _ = self._step(batch, rdrop=True)
        n = batch["input_ids"].shape[0]
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=n)
        self.log("train_main_loss", main_loss, on_epoch=True, batch_size=n)
        self.log("train_rdrop_kl", kl, on_epoch=True, batch_size=n)
        if aux_loss is not None:
            self.log("train_aux_loss", aux_loss, on_epoch=True, batch_size=n)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, _, proba, label = self._step(batch, rdrop=False)
        log_val_metrics(self, loss, proba, label, batch["input_ids"].shape[0])

    def test_step(self, batch, batch_idx):
        loss, _, _, _, proba, label = self._step(batch, rdrop=False)
        n = batch["input_ids"].shape[0]
        self.log("test_loss", loss, on_epoch=True, batch_size=n)
        self.metrics.test_ap(proba, label)
        self.log("test_ap", self.metrics.test_ap, on_epoch=True)
        proba_2d = torch.cat([1 - proba, proba], dim=1)
        self.metrics.test_f1_macro(proba_2d, label.squeeze(1))
        self.log("test_f1_macro", self.metrics.test_f1_macro, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Devolve ``{"video_ids", "proba"}`` — formato esperado por ``src/eval/protocol.py``."""
        _, _, _, _, proba, _ = self._step(batch, rdrop=False)
        return {"video_ids": batch.get("video_id"), "proba": proba.squeeze(1)}

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        max_epochs = self.trainer.max_epochs if self.trainer is not None else 200
        return configure_adamw_scheduler(
            trainable,
            lr=self.lr,
            weight_decay=self.weight_decay,
            monitor="val_loss",
            scheduler=self.scheduler,
            max_epochs=max_epochs,
            warmup_epochs=self.warmup_epochs,
        )


class TextFinetune:
    """Fachada registrada no Hydra (``model=text_finetune``, family="lightning").

    Lê ``configs/model/text_finetune.yaml``, baixa o backbone HuggingFace e
    monta ``TextFinetuneNet``/``LitTextFinetune``.
    """

    def __init__(self, cfg) -> None:
        self.model_name = str(cfg.get("model_name", DEFAULT_MODEL_NAME))
        self.num_frozen_layers = int(cfg.get("num_frozen_layers", 4))
        self.max_length = int(cfg.get("max_length", 256))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.head_hidden = int(cfg.get("head_hidden", 256))
        self.dim_tab = int(cfg.get("dim_tab") or 0)
        self.aux_loss_weight = float(cfg.get("aux_loss_weight", 0.3))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.1))
        self.rdrop_alpha = float(cfg.get("rdrop_alpha", 4.0))
        self.pos_weight = cfg.get("pos_weight", "auto")
        self.lr = float(cfg.get("lr", 2e-5))
        self.weight_decay = float(cfg.get("weight_decay", 1e-2))
        self.scheduler = str(cfg.get("scheduler", "cosine_warmup"))
        self.warmup_epochs = int(cfg.get("warmup_epochs", 2))

    @classmethod
    def from_config(cls, config) -> TextFinetune:
        return cls(cfg=config)

    def _load_backbone(self) -> tuple[nn.Module, int]:
        from transformers import AutoModel
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        log.info(f"Carregando backbone de texto: {self.model_name}")
        backbone = AutoModel.from_pretrained(self.model_name)
        freeze_backbone_layers(backbone, self.num_frozen_layers)
        return backbone, int(backbone.config.hidden_size)

    def build_module(self) -> TextFinetuneNet:
        """Instancia ``TextFinetuneNet`` com o backbone real (baixa da HF Hub)."""
        backbone, hidden_size = self._load_backbone()
        return TextFinetuneNet(
            backbone=backbone,
            hidden_size=hidden_size,
            dropout=self.dropout,
            head_hidden=self.head_hidden,
            dim_tab=self.dim_tab,
        )

    def build_lightning_module(self) -> LitTextFinetune:
        """Monta o ``LitTextFinetune`` pronto para ``L.Trainer.fit``."""
        pos_weight = None if self.pos_weight in ("auto", None) else float(self.pos_weight)
        return LitTextFinetune(
            net=self.build_module(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            pos_weight=pos_weight,
            label_smoothing=self.label_smoothing,
            rdrop_alpha=self.rdrop_alpha,
            aux_loss_weight=self.aux_loss_weight,
            scheduler=self.scheduler,
            warmup_epochs=self.warmup_epochs,
        )
