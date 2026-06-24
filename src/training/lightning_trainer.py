"""LightningTrainer — wrapper de L.Trainer + W&B (import LAZY de lightning).

Contrato ``BaseTrainer`` (README §6.4). Todo torch/lightning é importado DENTRO
dos métodos. Fluxo:
  1. Mapeia ``resolve_device(cfg.device)`` -> accelerator (cpu→"cpu", mps→"mps",
     cuda→"gpu", devices=1) e exporta ``PYTORCH_ENABLE_MPS_FALLBACK=1``.
  2. Monta ``L.Trainer`` (WandbLogger, EarlyStopping, ModelCheckpoint).
  3. ``fit``: treina o ``LitCrossAttention`` sobre ``VideoSequenceDataset`` (FASE 2).
  4. Calibra o limiar do sigmoid na val (max Macro-F1) e avalia a nível de vídeo
     (torchmetrics no LightningModule; agregação ``identity`` p/ relatório sklearn).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.base.trainer import BaseTrainer
from src.logger import get_logger
from src.training.aggregation import aggregate_to_video, calibrate_threshold
from src.training.metrics import evaluate_video_predictions

log = get_logger("training.lightning_trainer")

# device -> accelerator do Lightning (README §3)
_ACCELERATOR = {"cpu": "cpu", "mps": "mps", "cuda": "gpu"}


class LightningTrainer(BaseTrainer):
    """Trainer neural para a cross-attention (Lightning, opcional).

    Attributes:
        model: ``CrossAttentionFusion`` (fachada; constrói o ``LitCrossAttention``).
        config: config com grupos ``trainer`` (lightning), ``aggregation``, ``wandb``.
        threshold_: limiar do sigmoid calibrado na validação.
    """

    family: str = "lightning"

    def __init__(self, model: Any, config: Any) -> None:
        super().__init__(model=model, cfg=config)
        # Atalho legível para o ``cfg`` (mantém o corpo dos métodos claro).
        self.config = config
        self.threshold_: float | None = None
        self.results: dict[str, Any] = {}
        self._lit_module = None
        self._trainer = None
        self._ckpt_path: str | None = None
        self._ckpt_cb = None

    def _cfg_block(self, name: str) -> dict[str, Any]:
        block = getattr(self.config, name, None)
        if block is None and isinstance(self.config, dict):
            block = self.config.get(name, {})
        if block is None:
            return {}
        return dict(block) if not isinstance(block, dict) else block

    # ==========================================================================
    # Montagem do L.Trainer (device modular + W&B)
    # ==========================================================================

    def _build_trainer(self):
        """Monta o ``L.Trainer`` com accelerator derivado do device + callbacks W&B."""
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import WandbLogger

        from src.conf import resolve_device  # FASE 1

        device = resolve_device(getattr(self.config, "device", "auto"))
        accelerator = _ACCELERATOR.get(device.type, "cpu")
        if accelerator == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        log.info(f"Device resolvido: {device} -> accelerator='{accelerator}', devices=1")

        tcfg = self._cfg_block("trainer")
        wcfg = self._cfg_block("wandb")
        wandb_logger = WandbLogger(
            project=wcfg.get("project", "abaw-ah"),
            mode=wcfg.get("mode", "online"),  # online|offline|disabled
            log_model=True,
        )
        ckpt = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        )
        self._ckpt_cb = ckpt

        return L.Trainer(
            max_epochs=tcfg.get("max_epochs", 200),
            accelerator=accelerator,
            devices=tcfg.get("devices", 1),
            gradient_clip_val=tcfg.get("gradient_clip_val", 1.0),
            logger=wandb_logger,
            callbacks=[EarlyStopping(monitor="val_loss", patience=tcfg.get("patience", 20)), ckpt],
        )

    # ==========================================================================
    # Contrato BaseTrainer (README §6.4)
    # ==========================================================================

    def fit(self, train_data, val_data) -> dict[str, Any]:
        """Treina o ``LitCrossAttention`` e calibra o limiar do sigmoid na val.

        ``train_data``/``val_data`` são ``DataLoader`` sobre ``VideoSequenceDataset``
        (FASE 2): batches com ``audio_seq``/``text_seq`` (B,T,D), ``key_padding_mask``
        (B,T) e ``label``/``video_id``.
        """
        import lightning as L

        L.seed_everything(getattr(self.config, "seed", 42))
        self._lit_module = self.model.build_lightning_module()
        self._trainer = self._build_trainer()

        log.info("=== Treino da cross-attention (Lightning) ===")
        self._trainer.fit(self._lit_module, train_dataloaders=train_data, val_dataloaders=val_data)
        self._ckpt_path = getattr(self._ckpt_cb, "best_model_path", None) or "best"

        log.info("=== Calibração do limiar do sigmoid na validação ===")
        val_ids, val_proba = self._infer(val_data)
        val_labels = self._labels_from_loader(val_data)
        threshold, _ = calibrate_threshold(
            val_proba=val_proba,
            val_video_ids=val_ids,
            val_video_labels=val_labels,
            method="identity",
        )
        self.threshold_ = threshold
        self.results["val"] = self._evaluate(val_ids, val_proba, val_labels)
        self.results["threshold"] = threshold
        return self.results

    def evaluate(self, data) -> dict[str, Any]:
        """Avalia a nível de vídeo (limiar calibrado) — métricas sklearn canônicas."""
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit() antes de evaluate().")
        ids, proba = self._infer(data)
        return self._evaluate(ids, proba, self._labels_from_loader(data))

    def predict(self, data) -> dict[str, int]:
        """Predição A/H a nível de vídeo: ``{video_id: pred 0/1}``."""
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit() antes de predict().")
        ids, proba = self._infer(data)
        return aggregate_to_video(proba, ids, method="identity", threshold=self.threshold_)

    def save(self, out_dir) -> None:
        """Salva o caminho do checkpoint + limiar (o peso fica no ckpt do Lightning)."""
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trainer_state.json").write_text(
            json.dumps({"threshold": self.threshold_, "ckpt_path": self._ckpt_path}, indent=2)
        )
        log.info(f"LightningTrainer salvo em: {out_dir} (ckpt={self._ckpt_path})")

    @classmethod
    def load(cls, out_dir, model: Any, config: Any) -> LightningTrainer:
        """Recarrega o estado (limiar + caminho do ckpt) para inferência."""
        import json

        state = json.loads((Path(out_dir) / "trainer_state.json").read_text())
        trainer = cls(model=model, config=config)
        trainer.threshold_ = state.get("threshold")
        trainer._ckpt_path = state.get("ckpt_path")
        return trainer

    # ==========================================================================
    # Internos (inferência por vídeo + avaliação)
    # ==========================================================================

    def _infer(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Roda ``predict_step`` e devolve ``(video_ids, proba)`` como numpy."""
        outputs = self._trainer.predict(self._lit_module, dataloaders=loader, ckpt_path=self._ckpt_path)
        ids: list[str] = []
        proba: list[float] = []
        for out in outputs:
            ids.extend(list(out["video_ids"]))
            proba.extend(out["proba"].detach().cpu().numpy().tolist())
        return np.asarray(ids), np.asarray(proba, dtype=np.float32)

    @staticmethod
    def _labels_from_loader(loader) -> dict[str, int]:
        """Extrai ``{video_id: video_label}`` do ``VideoSequenceDataset`` do loader."""
        ds = loader.dataset
        # ds.video_labels é o acessor público {video_id: video_label} do
        # VideoSequenceDataset (FASE_2); alinhado com WindowMatrixView.
        return {str(vid): int(lab) for vid, lab in ds.video_labels.items()}

    def _evaluate(self, ids, proba, labels) -> dict[str, Any]:
        preds = aggregate_to_video(proba, ids, method="identity", threshold=self.threshold_)
        scores = {str(v): float(p) for v, p in zip(ids, proba)}
        return evaluate_video_predictions(video_labels=labels, video_pred=preds, video_score=scores)
