"""MoE Fusion — fusão ancorada em texto + Mixture-of-Experts (``references/improvement_plan.md``).

Implementa o "CENTER" da figura do plano, sobre ramos por modalidade configuráveis:

    ramos (LEFT):  h_m = encoder_m(entrada_m)            ← src/models/encoders.py
    âncora:        b   = d_text(h_text)
    resíduos:      z   = LN(b + Σ_m g_m · d_m(h_m)),   g_m = σ(MLP[b; d_m(h_m)])
    cabeça:        e   = SoftMoE([z ‖ h_lateral...])   (K experts; ou MLP)
                   p(A/H) = σ(Linear(e))

``e`` é o vetor pré-logit exportado em ``eval_<split>/embeddings.npy`` (entrada do
MoERouter). Cada ramo declara um ``role``:

- ``anchor`` — exatamente um (o texto): ``b``;
- ``residual`` — ajuste residual gateado (áudio, ASR timing, rosto, tabular, cena);
- ``head`` — entra direto na cabeça, concatenado a ``z`` (ex.: os 11 marcadores de
  hesitação, a "cabeça auxiliar" tracejada da figura).

Perdas: BCE principal (``pos_weight`` auto, label smoothing) + R-Drop opcional + cabeças
auxiliares por ramo (``aux_weight``, estilo AMF: o ramo também prediz A/H sozinho) +
supervisão por janela (``window_weight``, rótulos de ``time_detailed_ah``) + balanceamento
dos experts só quando um passa de ``balance_threshold`` do tráfego.

Um ramo só (sem resíduos) é o membro UNIMODAL do mesmo modelo: o preset de texto fine-tune
(Rodada 1), o de áudio temporal e o de ASR timing são esta mesma classe com um ramo.

Importado só via registro lazy (``registry.py``) — o ``torch`` no topo não toca o caminho RF.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from src.logger import get_logger
from src.models.blocks import ProjectionBlock, ReliabilityGate, SoftMoE, load_balance_loss
from src.models.encoders import build_encoder
from src.models.lightning_utils import (
    bce_with_logits,
    build_classification_metrics,
    configure_adamw_scheduler,
    log_val_metrics,
    smooth_binary_labels,
    symmetric_kl_from_logits,
)

log = get_logger("models.moe_fusion")

ROLES = ("anchor", "residual", "head")


# =============================================================================
# nn.Module — fusão ancorada em texto + cabeça MoE
# =============================================================================


class TextAnchoredMoE(nn.Module):
    """Ramos por modalidade → fusão residual gateada sobre a âncora → cabeça SoftMoE.

    Args:
        branches: ``{nome: spec}`` (bloco ``model.branches`` do YAML).
        input_dims: ``{coluna: d}`` das colunas por janela (inferido do dataset).
        dim: largura da fusão (``b``, ``d_m(h_m)``, ``z``).
        head: ``moe`` (SoftMoE) | ``mlp``.
        num_experts / expert_hidden: experts da cabeça MoE.
        gate / gate_init_bias: porta escalar|vetorial e o viés inicial (``< 0`` = fechada).
        modality_dropout: prob. de zerar a porta de um ramo residual por amostra no treino.
    """

    def __init__(
        self,
        branches: dict[str, dict[str, Any]],
        input_dims: dict[str, int],
        dim: int = 256,
        head: str = "moe",
        num_experts: int = 4,
        expert_hidden: int | None = None,
        dropout: float = 0.1,
        gate: str = "scalar",
        gate_init_bias: float = 0.0,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.specs = {name: dict(spec) for name, spec in branches.items()}
        by_role = {
            r: [n for n, s in self.specs.items() if s.get("role", "residual") == r] for r in ROLES
        }
        bad = [n for n, s in self.specs.items() if s.get("role", "residual") not in ROLES]
        if bad or len(by_role["anchor"]) != 1:
            raise ValueError(
                f"Ramos precisam de exatamente 1 role=anchor (roles: {ROLES}); {self.specs}"
            )
        self.anchor = by_role["anchor"][0]
        self.residual = by_role["residual"]
        self.side = by_role["head"]
        self.modality_dropout = float(modality_dropout)

        self.encoders = nn.ModuleDict(
            {
                n: build_encoder(s, input_dims.get(s.get("column", "")))
                for n, s in self.specs.items()
            }
        )
        self.proj = nn.ModuleDict(
            {
                n: ProjectionBlock(self.encoders[n].out_dim, dim, dropout)
                for n in [self.anchor, *self.residual]
            }
        )
        self.gates = nn.ModuleDict(
            {n: ReliabilityGate(dim, gate, gate_init_bias) for n in self.residual}
        )
        self.fuse_norm = nn.LayerNorm(dim)

        head_in = dim + sum(self.encoders[n].out_dim for n in self.side)
        self.head_kind = head
        if head == "moe":
            self.moe = SoftMoE(head_in, dim, num_experts, hidden=expert_hidden, dropout=dropout)
            self.head_out = nn.Dropout(dropout)
        elif head == "mlp":
            self.head_out = nn.Sequential(nn.Linear(head_in, dim), nn.GELU(), nn.Dropout(dropout))
        else:
            raise ValueError(f"head desconhecida: {head!r} (moe|mlp)")
        self.classifier = nn.Linear(dim, 1)
        self.aux = nn.ModuleDict(
            {
                n: nn.Linear(self.encoders[n].out_dim, 1)
                for n, s in self.specs.items()
                if float(s.get("aux_weight", 0.0) or 0.0) > 0
            }
        )

    def _encode(self, name: str, batch: dict[str, Any]):
        spec = self.specs[name]
        if spec.get("encoder") == "hf_text":
            return self.encoders[name](batch["input_ids"], batch["attention_mask"])
        return self.encoders[name](batch["features"][spec["column"]], batch["key_padding_mask"])

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Batch do ``collate_sequences`` → logit ``(B, 1)`` + saídas auxiliares/telemetria."""
        h: dict[str, torch.Tensor] = {}
        window_logits: dict[str, torch.Tensor] = {}
        routing: dict[str, torch.Tensor] = {}
        for name in self.encoders:
            h[name], wl, mix_w = self._encode(name, batch)
            if wl is not None:
                window_logits[name] = wl
            if mix_w is not None:
                routing[name] = mix_w

        b = self.proj[self.anchor](h[self.anchor])
        z = b
        gates: dict[str, torch.Tensor] = {}
        for name in self.residual:
            d = self.proj[name](h[name])
            g = self.gates[name](b, d)
            if self.training and self.modality_dropout > 0:
                keep = torch.rand(g.shape[0], 1, device=g.device) >= self.modality_dropout
                g = g * keep.to(g.dtype)
            gates[name] = g
            z = z + g * d
        z = self.fuse_norm(z)

        head_in = torch.cat([z, *(h[n] for n in self.side)], dim=-1)
        if self.head_kind == "moe":
            mixed, routing["head"] = self.moe(head_in)
            embedding = self.head_out(mixed)
        else:
            embedding = self.head_out(head_in)
        return {
            "logit": self.classifier(embedding),
            "embedding": embedding,
            "gates": gates,
            "routing": routing,
            "aux_logits": {n: head(h[n]) for n, head in self.aux.items()},
            "window_logits": window_logits,
        }


# =============================================================================
# LightningModule — perdas + métricas + otimizador
# =============================================================================


class LitMoEFusion(L.LightningModule):
    """Treina o :class:`TextAnchoredMoE` (1 logit por vídeo, métricas iguais às do CA)."""

    def __init__(
        self,
        net: TextAnchoredMoE,
        *,
        lr: float = 1e-3,
        lr_backbone: float = 2e-5,
        weight_decay: float = 1e-2,
        pos_weight: float | None = None,
        label_smoothing: float = 0.0,
        rdrop_alpha: float = 0.0,
        balance_weight: float = 0.0,
        balance_threshold: float = 0.7,
        scheduler: str = "cosine_warmup",
        warmup_epochs: int = 1,
        trainer_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.metrics = build_classification_metrics()
        self.lr, self.lr_backbone, self.weight_decay = lr, lr_backbone, weight_decay
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.rdrop_alpha = rdrop_alpha
        self.balance_weight, self.balance_threshold = balance_weight, balance_threshold
        self.scheduler, self.warmup_epochs = scheduler, warmup_epochs
        self.trainer_cfg = dict(trainer_cfg or {})
        self.aux_weights = {n: float(net.specs[n]["aux_weight"]) for n in net.aux}
        self.window_weights = {
            n: float(s["window_weight"])
            for n, s in net.specs.items()
            if float(s.get("window_weight", 0.0) or 0.0) > 0
        }

    # ---- perdas -------------------------------------------------------------
    def _main_loss(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        target = smooth_binary_labels(label, self.label_smoothing)
        return bce_with_logits(logit, target, self.pos_weight)

    def _losses(self, out: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        label = batch["label"].float()  # (B, 1)
        parts = {"main": self._main_loss(out["logit"], label)}
        for name, w in self.aux_weights.items():
            parts[f"aux_{name}"] = w * F.binary_cross_entropy_with_logits(
                out["aux_logits"][name], label
            )
        win_label = batch.get("window_label")
        for name, w in self.window_weights.items():
            valid = win_label >= 0  # -1 = sem rótulo de janela (test) ou padding
            if valid.any():
                logits = out["window_logits"][name][valid]
                parts[f"window_{name}"] = w * F.binary_cross_entropy_with_logits(
                    logits, win_label[valid].float()
                )
        if self.balance_weight > 0:
            for name, weights in out["routing"].items():
                parts[f"balance_{name}"] = self.balance_weight * load_balance_loss(
                    weights, self.balance_threshold
                )
        return parts

    # ---- passos -------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        out = self.net(batch)
        parts = self._losses(out, batch)
        loss = sum(parts.values())
        if self.rdrop_alpha > 0:  # 2º passe com dropout independente + consistência
            out2 = self.net(batch)
            main2 = self._main_loss(out2["logit"], batch["label"].float())
            parts["rdrop_kl"] = symmetric_kl_from_logits(out["logit"], out2["logit"])
            loss = loss + 0.5 * (main2 - parts["main"]) + self.rdrop_alpha * parts["rdrop_kl"]
        n = batch["label"].shape[0]
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=n)
        for key, value in parts.items():
            self.log(f"train_{key}", value, on_step=False, on_epoch=True, batch_size=n)
        self._log_telemetry(out, "train", n)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.net(batch)
        label = batch["label"].float()
        loss = self._main_loss(out["logit"], label)
        log_val_metrics(self, loss, torch.sigmoid(out["logit"]), label.int(), label.shape[0])
        self._log_telemetry(out, "val", label.shape[0])

    def test_step(self, batch, batch_idx):
        out = self.net(batch)
        label = batch["label"].float()
        proba = torch.sigmoid(out["logit"])
        self.log("test_loss", self._main_loss(out["logit"], label), on_epoch=True)
        self.metrics.test_ap(proba, label.int())
        self.log("test_ap", self.metrics.test_ap, on_epoch=True)
        self.metrics.test_f1_macro(torch.cat([1 - proba, proba], dim=1), label.int().squeeze(1))
        self.log("test_f1_macro", self.metrics.test_f1_macro, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """``video_ids``/``proba`` + vetor pré-logit, uso dos experts e portas por amostra."""
        out = self.net(batch)
        pred = {
            "video_ids": batch["video_id"],
            "proba": torch.sigmoid(out["logit"]).squeeze(1),
            "embedding": out["embedding"],
        }
        if "head" in out["routing"]:
            pred["router_weights"] = out["routing"]["head"]
        if out["gates"]:
            pred["gates"] = torch.stack([g.mean(dim=-1) for g in out["gates"].values()], dim=1)
        return pred

    def _log_telemetry(self, out: dict[str, Any], stage: str, n: int) -> None:
        """Uso máximo de um expert (regra dos 70%) e abertura média de cada porta."""
        if "head" in out["routing"]:
            usage = out["routing"]["head"].mean(dim=0)
            self.log(
                f"{stage}_moe_max_usage", usage.max(), on_step=False, on_epoch=True, batch_size=n
            )
        for name, g in out["gates"].items():
            self.log(f"{stage}_gate_{name}", g.mean(), on_step=False, on_epoch=True, batch_size=n)

    # ---- otimização ---------------------------------------------------------
    def configure_optimizers(self):
        backbone = [
            p
            for enc in self.net.encoders.values()
            if getattr(enc, "is_backbone", False)
            for p in enc.parameters()
            if p.requires_grad
        ]
        ids = {id(p) for p in backbone}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        groups = [{"params": rest, "lr": self.lr}]
        if backbone:
            groups.append({"params": backbone, "lr": self.lr_backbone})
        max_epochs = self.trainer.max_epochs if self.trainer is not None else 100
        return configure_adamw_scheduler(
            groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            monitor=str(self.trainer_cfg.get("monitor", "val_ap")),
            mode=str(self.trainer_cfg.get("mode", "max")),
            scheduler=self.scheduler,
            max_epochs=max_epochs,
            warmup_epochs=self.warmup_epochs,
        )


# =============================================================================
# Fachada registrável (family="lightning")
# =============================================================================


class MoEFusion:
    """Fachada do ``moe_fusion`` para o registry (``configs/model/moe_fusion.yaml``).

    O ``LightningTrainer`` chama :meth:`infer_dims` (dimensões das colunas no cache),
    :meth:`build_lightning_module` e :meth:`init_weights` (só no treino); os datasets pedem
    as entradas via :meth:`data_spec`.
    ``input_dims`` vai no ``trainer_state.json`` (``model_cfg``) e é restaurado no load.
    """

    def __init__(self, cfg: Any) -> None:
        c = _plain(cfg)
        self.branches: dict[str, dict[str, Any]] = {n: dict(s) for n, s in c["branches"].items()}
        self.fusion: dict[str, Any] = dict(c.get("fusion") or {})
        self.optim: dict[str, Any] = dict(c.get("optim") or {})
        self.loss: dict[str, Any] = dict(c.get("loss") or {})
        self.input_dims: dict[str, int] = dict(c.get("input_dims") or {})
        self.pos_weight = self.loss.get("pos_weight", "auto")  # resolvido pelo trainer
        self._resolved_pos_weight: float | None = None

    @classmethod
    def from_config(cls, config: Any) -> MoEFusion:
        return cls(cfg=config)

    @staticmethod
    def data_spec(model_cfg: Any) -> dict[str, Any]:
        """Entradas que o dataset precisa montar: colunas por janela e transcrição tokenizada."""
        branches = _plain(model_cfg)["branches"]
        columns = sorted({s["column"] for s in branches.values() if s.get("encoder") != "hf_text"})
        text = next((s for s in branches.values() if s.get("encoder") == "hf_text"), None)
        transcript = (
            {"model_name": text["model_name"], "max_length": int(text.get("max_length", 256))}
            if text
            else None
        )
        return {"columns": columns, "transcript": transcript}

    def infer_dims(self, dataset: Any) -> None:
        """Lê do dataset a dimensão de cada coluna usada pelos ramos."""
        self.input_dims = {c: int(dataset.dims[c]) for c in self.data_spec(self._cfg())["columns"]}
        log.info(f"Dims das colunas (do cache): {self.input_dims}")

    def build_module(self) -> TextAnchoredMoE:
        return TextAnchoredMoE(self.branches, self.input_dims, **self.fusion)

    def init_weights(self, lit_module: LitMoEFusion) -> None:
        """Gancho de TREINO (``LightningTrainer.fit``): ramos com ``init_from`` partem dos
        pesos do mesmo ramo de outro run. Não roda no load (os pesos já estão no ``.ckpt``)."""
        for name, spec in self.branches.items():
            if spec.get("init_from"):
                _load_branch_weights(lit_module.net, name, str(spec["init_from"]))

    def build_lightning_module(self, trainer_cfg: dict[str, Any] | None = None) -> LitMoEFusion:
        o, lo = self.optim, self.loss
        return LitMoEFusion(
            self.build_module(),
            lr=float(o.get("lr", 1e-3)),
            lr_backbone=float(o.get("lr_backbone", 2e-5)),
            weight_decay=float(o.get("weight_decay", 1e-2)),
            scheduler=str(o.get("scheduler", "cosine_warmup")),
            warmup_epochs=int(o.get("warmup_epochs", 1)),
            pos_weight=self._resolved_pos_weight,
            label_smoothing=float(lo.get("label_smoothing", 0.0)),
            rdrop_alpha=float(lo.get("rdrop_alpha", 0.0)),
            balance_weight=float(lo.get("balance_weight", 0.0)),
            balance_threshold=float(lo.get("balance_threshold", 0.7)),
            trainer_cfg=trainer_cfg,
        )

    def _cfg(self) -> dict[str, Any]:
        return {"branches": self.branches}


def _load_branch_weights(net: TextAnchoredMoE, branch: str, run: str) -> None:
    """Inicializa o encoder ``branch`` com os pesos do MESMO ramo de outro run (``init_from``).

    Ex.: o MoE da Rodada 2 parte do texto fine-tunado da Rodada 1 (no OOF, o run do mesmo
    fold: ``init_from: outputs/oof/<run>/fold{fold}``).
    """
    from src.models.checkpoint_compat import load_state_dict_from_ckpt, resolve_ckpt_path

    ckpt = resolve_ckpt_path(run)
    if ckpt is None:
        raise FileNotFoundError(f"init_from do ramo '{branch}': checkpoint não encontrado em {run}")
    prefix = f"net.encoders.{branch}."
    state = {
        k[len(prefix) :]: v
        for k, v in load_state_dict_from_ckpt(ckpt).items()
        if k.startswith(prefix)
    }
    if not state:
        raise KeyError(f"init_from: {ckpt} não tem pesos do ramo '{branch}' ({prefix}*)")
    net.encoders[branch].load_state_dict(state)
    log.info(f"Ramo '{branch}' inicializado de {ckpt} ({len(state)} tensores).")


def _plain(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
