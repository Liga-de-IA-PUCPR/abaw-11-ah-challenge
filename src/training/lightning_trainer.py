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
        # Definido por main._run_train ANTES do fit → o ModelCheckpoint grava o .ckpt
        # aqui (junto do trainer_state.json), unificando o run dir p/ evaluate/submit.
        self.output_dir: str | None = None

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
        # save_dir sob output_root (gitignored) — mantém logs/ckpts fora da raiz do repo.
        try:
            out_root = str(self.config.data.paths.output_root)
        except Exception:  # noqa: BLE001
            out_root = "outputs"
        wandb_logger = WandbLogger(
            project=wcfg.get("project", "abaw-ah"),
            mode=wcfg.get("mode", "online"),  # online|offline|disabled
            save_dir=out_root,
            log_model=True,
        )
        # dirpath = <run_dir>/checkpoints → o .ckpt fica no MESMO run dir do
        # trainer_state.json (resolve_latest_checkpoint acha os dois juntos).
        ckpt_dir = str(Path(self.output_dir) / "checkpoints") if self.output_dir else None
        ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
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
        # Infere as dims dos embeddings do CACHE (librosa 320 / wav2vec2 768) em vez de
        # confiar no hardcode da config — assim o modelo casa com o Parquet existente.
        d_a, d_b, d_tab = self._dims_from_loader(train_data)
        if d_a and d_b:
            self.model.dim_a, self.model.dim_b = d_a, d_b
            log.info(f"Dims inferidas do cache: dim_a={d_a}, dim_b={d_b}")
        # dim_tab só importa quando o ramo tabular está ligado (model.use_tabular).
        if getattr(self.model, "use_tabular", False) and d_tab:
            self.model.dim_tab = d_tab
            log.info(f"Ramo tabular ligado: dim_tab={d_tab} (funde tab_seq na cross-attention)")
        self._lit_module = self.model.build_lightning_module()
        self._trainer = self._build_trainer()

        log.info("=== Treino da cross-attention (Lightning) ===")
        self._trainer.fit(self._lit_module, train_dataloaders=train_data, val_dataloaders=val_data)
        self._ckpt_path = getattr(self._ckpt_cb, "best_model_path", None) or "best"

        log.info("=== Calibração do limiar do sigmoid na validação ===")
        val_ids, val_proba = self._infer(val_data)
        val_labels = self._labels_from_loader(val_data)
        # Honra aggregation.threshold: "auto" calibra na val (max métrica); um float
        # FIXA o limiar e pula a calibração — mesma semântica do SklearnTrainer.
        agg = self._cfg_block("aggregation")
        thr_setting = agg.get("threshold", "auto")
        if thr_setting == "auto":
            threshold, _ = calibrate_threshold(
                val_proba=val_proba,
                val_video_ids=val_ids,
                val_video_labels=val_labels,
                method="identity",
                metric=self._cfg_block("metrics").get("primary", "macro_f1"),
                selection=agg.get("calibration", "smooth"),
                smooth_window=float(agg.get("smooth_window", 0.10)),
            )
        else:
            threshold = float(thr_setting)
            log.info(f"Limiar fixo da config: {threshold:.3f}")
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

    def video_outputs(self, data) -> dict[str, np.ndarray]:
        """Arrays a nível de vídeo p/ plots/relatórios (FASE 5): ids, y_true, y_proba, y_pred."""
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit()/load() antes.")
        ids, proba = self._infer(data)
        labels = self._labels_from_loader(data)
        sel = [i for i in range(len(ids)) if str(ids[i]) in labels]
        y_true = np.array([labels[str(ids[i])] for i in sel], dtype=np.int64)
        y_proba = np.array([float(proba[i]) for i in sel], dtype=np.float32)
        y_pred = (y_proba >= self.threshold_).astype(np.int64)
        return {
            "video_ids": np.asarray([ids[i] for i in sel]),
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }

    def save(self, out_dir) -> None:
        """Salva o caminho do checkpoint + limiar (o peso fica no ckpt do Lightning)."""
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    "threshold": self.threshold_,
                    "ckpt_path": self._ckpt_path,
                    "dim_a": int(self.model.dim_a),  # dims treinadas → load reconstrói igual
                    "dim_b": int(self.model.dim_b),
                    "dim_tab": int(getattr(self.model, "dim_tab", 0)),
                    "use_tabular": bool(getattr(self.model, "use_tabular", False)),
                },
                indent=2,
            )
        )
        log.info(f"LightningTrainer salvo em: {out_dir} (ckpt={self._ckpt_path})")

    @classmethod
    def load(cls, out_dir, model: Any, config: Any) -> LightningTrainer:
        """Recarrega o estado (limiar + caminho do ckpt) para inferência."""
        import json

        import lightning as L

        from src.conf import resolve_device

        state = json.loads((Path(out_dir) / "trainer_state.json").read_text())
        trainer = cls(model=model, config=config)
        trainer.threshold_ = state.get("threshold")
        trainer.output_dir = str(out_dir)
        # ckpt_path do estado; se sumiu (run dir movido), procura o .ckpt DENTRO do
        # próprio run dir → checkpoint auto-contido e portátil.
        ckpt_path = state.get("ckpt_path")
        if not ckpt_path or not Path(ckpt_path).exists():
            cands = sorted(Path(out_dir).glob("checkpoints/*.ckpt")) + sorted(
                Path(out_dir).glob("*.ckpt")
            )
            ckpt_path = str(cands[-1]) if cands else ckpt_path
        trainer._ckpt_path = ckpt_path
        # Restaura as dims treinadas p/ o módulo casar com os pesos do ckpt, e monta
        # um L.Trainer leve (sem logger/callbacks) p/ inferência (evaluate/predict).
        if state.get("dim_a") and state.get("dim_b"):
            model.dim_a, model.dim_b = int(state["dim_a"]), int(state["dim_b"])
        # Restaura o ramo tabular exatamente como no treino (senão o state_dict não casa).
        if "use_tabular" in state:
            model.use_tabular = bool(state["use_tabular"])
        if state.get("dim_tab"):
            model.dim_tab = int(state["dim_tab"])
        trainer._lit_module = model.build_lightning_module()
        device = resolve_device(getattr(config, "device", "auto"))
        accelerator = _ACCELERATOR.get(device.type, "cpu")
        if accelerator == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        trainer._trainer = L.Trainer(
            accelerator=accelerator, devices=1, logger=False, enable_progress_bar=False
        )
        return trainer

    # ==========================================================================
    # Internos (inferência por vídeo + avaliação)
    # ==========================================================================

    def _infer(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """Roda ``predict_step`` e devolve ``(video_ids, proba)`` como numpy."""
        outputs = self._trainer.predict(
            self._lit_module, dataloaders=loader, ckpt_path=self._ckpt_path
        )
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

    @staticmethod
    def _dims_from_loader(loader) -> tuple[int, int, int]:
        """Dims ``(d_audio, d_text, d_tab)`` do ``VideoSequenceDataset`` (0 se ausente)."""
        ds = loader.dataset
        return (
            int(getattr(ds, "dim_audio", 0)),
            int(getattr(ds, "dim_text", 0)),
            int(getattr(ds, "dim_tab", 0)),
        )

    def _evaluate(self, ids, proba, labels) -> dict[str, Any]:
        preds = aggregate_to_video(proba, ids, method="identity", threshold=self.threshold_)
        scores = {str(v): float(p) for v, p in zip(ids, proba, strict=False)}
        return evaluate_video_predictions(video_labels=labels, video_pred=preds, video_score=scores)
