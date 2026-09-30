"""Trainer para pré-treino HeteroGAE (reconstrução de arestas, sem labels)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.base.trainer import BaseTrainer
from src.logger import get_logger

log = get_logger("training.gae_pretrain_trainer")

_ACCELERATOR = {"cpu": "cpu", "mps": "mps", "cuda": "gpu"}


class GaePretrainTrainer(BaseTrainer):
    """Treina HeteroGAE e salva pesos do encoder para init do HeteroGAT."""

    family: str = "lightning"

    def __init__(self, model: Any, config: Any) -> None:
        super().__init__(model=model, cfg=config)
        self.config = config
        self.results: dict[str, Any] = {}
        self._lit_module = None
        self._trainer = None
        self._ckpt_path: str | None = None
        self._ckpt_cb = None
        self.output_dir: str | None = None

    def _cfg_block(self, name: str) -> dict[str, Any]:
        block = getattr(self.config, name, None)
        if block is None and isinstance(self.config, dict):
            block = self.config.get(name, {})
        if block is None:
            return {}
        return dict(block) if not isinstance(block, dict) else block

    def _build_trainer(self):
        import lightning as L
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

        from src.conf import resolve_device

        device = resolve_device(getattr(self.config, "device", "auto"))
        accelerator = _ACCELERATOR.get(device.type, "cpu")
        if accelerator == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        tcfg = self._cfg_block("trainer")
        ckpt_dir = str(Path(self.output_dir) / "checkpoints") if self.output_dir else None
        ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_recon_loss",
            mode="min",
            save_top_k=1,
            filename="gae-best-{epoch:02d}-{val_recon_loss:.4f}",
        )
        self._ckpt_cb = ckpt

        return L.Trainer(
            max_epochs=tcfg.get("max_epochs", 50),
            accelerator=accelerator,
            devices=tcfg.get("devices", 1),
            gradient_clip_val=tcfg.get("gradient_clip_val", 1.0),
            logger=False,
            callbacks=[
                EarlyStopping(monitor="val_recon_loss", patience=tcfg.get("patience", 10)),
                ckpt,
            ],
        )

    def fit(self, train_data, val_data) -> dict[str, Any]:
        import lightning as L

        L.seed_everything(getattr(self.config, "seed", 42))
        d_a, d_b, d_tab = self._dims_from_loader(train_data)
        if d_a and d_b:
            self.model.dim_a, self.model.dim_b = d_a, d_b
        if d_tab:
            self.model.dim_tab = d_tab

        self._lit_module = self.model.build_lightning_module()
        self._trainer = self._build_trainer()
        log.info("=== Pré-treino HeteroGAE (recon_loss) ===")
        self._trainer.fit(self._lit_module, train_dataloaders=train_data, val_dataloaders=val_data)
        self._ckpt_path = getattr(self._ckpt_cb, "best_model_path", None)
        return self.results

    def evaluate(self, data) -> dict[str, Any]:
        raise NotImplementedError("GaePretrainTrainer não suporta evaluate.")

    def predict(self, data) -> dict[str, int]:
        raise NotImplementedError("GaePretrainTrainer não suporta predict.")

    def save(self, out_dir) -> None:
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        encoder_path = out_dir / "gae_encoder.pt"
        self.model.save_encoder(self._lit_module, encoder_path)
        (out_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    "ckpt_path": self._ckpt_path,
                    "gae_encoder": str(encoder_path),
                    "dim_a": int(self.model.dim_a),
                    "dim_b": int(self.model.dim_b),
                    "dim_tab": int(self.model.dim_tab),
                },
                indent=2,
            )
        )
        log.info(f"GaePretrainTrainer salvo em: {out_dir}")

    @classmethod
    def load(cls, out_dir, model: Any, config: Any) -> GaePretrainTrainer:
        """Recarrega metadados do pré-treino (encoder path)."""
        import json

        trainer = cls(model=model, config=config)
        state_path = Path(out_dir) / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            enc = state.get("gae_encoder")
            if enc and hasattr(model, "_gae_init_path"):
                model._gae_init_path = enc
        trainer.output_dir = str(out_dir)
        return trainer

    @staticmethod
    def _dims_from_loader(loader) -> tuple[int, int, int]:
        ds = loader.dataset
        return (
            int(getattr(ds, "dim_audio", 0)),
            int(getattr(ds, "dim_text", 0)),
            int(getattr(ds, "dim_tab", 0)),
        )
