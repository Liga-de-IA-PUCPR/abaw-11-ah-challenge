"""LightningTrainer — wrapper de L.Trainer + W&B (import LAZY de lightning).

Contrato ``BaseTrainer`` (README §6.4). Todo torch/lightning é importado DENTRO
dos métodos. Fluxo:
  1. Mapeia ``resolve_device(cfg.device)`` -> accelerator (cpu→"cpu", mps→"mps",
     cuda→"gpu", devices=1) e exporta ``PYTORCH_ENABLE_MPS_FALLBACK=1``.
  2. Monta ``L.Trainer`` (WandbLogger, EarlyStopping, ModelCheckpoint; opcionais:
     ``accumulate_grad_batches``, SWA, ``precision``).
  3. ``fit``: treina o LightningModule do modelo (cross-attention, GNNs heterogêneos,
     GNN facial) sobre ``VideoSequenceDataset`` (FASE 2).
  4. Calibra o limiar do sigmoid na val (max Macro-F1) e avalia a nível de vídeo
     (torchmetrics no LightningModule; agregação ``identity`` p/ relatório sklearn).

Qualquer fachada de modelo registrada como ``family="lightning"`` serve, desde que exponha
``build_lightning_module()`` (opcionalmente ``build_lightning_module(trainer_cfg=...)``) e
os atributos de dimensão ``dim_a``/``dim_b`` (e ``dim_tab``, se usar o ramo tabular). Gancho
opcional do modelo: ``pos_weight`` (``auto`` = neg/pos do treino).
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
    """Trainer neural dos modelos da família ``lightning`` (opcional).

    Attributes:
        model: fachada do modelo (ex.: ``CrossAttentionFusion``, ``HeteroGnnFusion``);
            constrói o ``LightningModule``.
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
        from lightning.pytorch.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            StochasticWeightAveraging,
        )
        from lightning.pytorch.loggers import WandbLogger

        from src.conf import resolve_device  # FASE 1

        device = resolve_device(getattr(self.config, "device", "auto"))
        accelerator = _ACCELERATOR.get(device.type, "cpu")
        if accelerator == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        log.info(f"Device resolvido: {device} -> accelerator='{accelerator}', devices=1")

        tcfg = self._cfg_block("trainer")
        wcfg = self._cfg_block("wandb")
        # save_dir = o run dir (<run>/wandb/, no OOF <run>/fold<k>/wandb/): cada run W&B fica
        # junto do run que o gerou. Sem run dir, sob output_root (gitignored).
        try:
            out_root = str(self.config.data.paths.output_root)
        except Exception:  # noqa: BLE001
            out_root = "outputs"
        save_dir = self.output_dir or out_root
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        try:  # nome do run W&B = o run dir (ex.: oof/moe-r1-text/<ts>/fold0)
            run_name = str(Path(save_dir).relative_to(out_root)) if self.output_dir else None
        except ValueError:
            run_name = None
        wandb_logger = WandbLogger(
            project=wcfg.get("project", "abaw-ah"),
            name=run_name,
            mode=wcfg.get("mode", "online"),  # online|offline|disabled
            save_dir=save_dir,
            log_model=True,
        )
        # Critério de seleção do checkpoint/early-stop. Default = val_ap (AP, livre de
        # limiar): a tarefa é "rankeia + calibra o limiar depois", então o melhor modelo
        # é o de melhor RANKING, não o de menor BCE (val_loss diverge do AP/F1 e tende a
        # escolher uma época sub-treinada). Configurável via trainer.monitor/mode.
        monitor = tcfg.get("monitor", "val_ap")
        mode = tcfg.get("mode", "max")
        # dirpath = <run_dir>/checkpoints → o .ckpt fica no MESMO run dir do
        # trainer_state.json (resolve_latest_checkpoint acha os dois juntos).
        ckpt_dir = str(Path(self.output_dir) / "checkpoints") if self.output_dir else None
        ckpt = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            filename=f"best-{{epoch:02d}}-{{{monitor}:.4f}}",
        )
        self._ckpt_cb = ckpt
        log.info(f"Seleção de checkpoint/early-stop: monitor='{monitor}' (mode='{mode}')")

        callbacks: list[Any] = [
            EarlyStopping(monitor=monitor, mode=mode, patience=tcfg.get("patience", 20)),
            ckpt,
        ]
        # SWA opcional (trainer.swa=true): média de pesos no fim do treino.
        if tcfg.get("swa"):
            max_ep = int(tcfg.get("max_epochs", 200))
            swa_start = tcfg.get("swa_epoch_start")
            swa_start = max(1, int(0.75 * max_ep)) if swa_start is None else int(swa_start)
            callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=float(tcfg.get("swa_lrs", 1e-5)),
                    swa_epoch_start=swa_start,
                    annealing_epochs=int(tcfg.get("swa_annealing_epochs", 5)),
                )
            )
            log.info(f"SWA ativo: início época {swa_start}, lr={tcfg.get('swa_lrs', 1e-5)}")

        # Opcionais (só repassados se definidos): batch efetivo maior com pouca VRAM
        # (accumulate_grad_batches), precisão mista (precision) e frequência de log.
        extra: dict[str, Any] = {}
        accumulate = int(tcfg.get("accumulate_grad_batches", 1) or 1)
        if accumulate > 1:
            extra["accumulate_grad_batches"] = accumulate
            log.info(
                f"Gradient accumulation: {accumulate} (batch efetivo = batch_size × {accumulate})"
            )
        for key in ("precision", "log_every_n_steps"):
            if tcfg.get(key) is not None:
                extra[key] = tcfg[key]

        return L.Trainer(
            max_epochs=tcfg.get("max_epochs", 200),
            accelerator=accelerator,
            devices=tcfg.get("devices", 1),
            gradient_clip_val=tcfg.get("gradient_clip_val", 1.0),
            logger=wandb_logger,
            callbacks=callbacks,
            **extra,
        )

    # ==========================================================================
    # Contrato BaseTrainer (README §6.4)
    # ==========================================================================

    def _build_lit_module(self):
        """Instancia o LightningModule; passa ``trainer_cfg`` só se o modelo aceitar.

        Os GNNs usam ``trainer_cfg`` (ex.: monitor/max_epochs no scheduler); a
        cross-attention não recebe nada (assinatura original preservada).
        """
        import inspect

        sig = inspect.signature(self.model.build_lightning_module)
        if "trainer_cfg" in sig.parameters:
            return self.model.build_lightning_module(trainer_cfg=self._cfg_block("trainer"))
        return self.model.build_lightning_module()

    def fit(self, train_data, val_data) -> dict[str, Any]:
        """Treina o LightningModule do modelo e calibra o limiar do sigmoid na val.

        ``train_data``/``val_data`` são ``DataLoader`` sobre ``VideoSequenceDataset``
        (FASE 2): batches com ``audio_seq``/``text_seq`` (B,T,D), ``key_padding_mask``
        (B,T) e ``label``/``video_id``.
        """
        import lightning as L

        L.seed_everything(getattr(self.config, "seed", 42))
        _set_cuda_matmul_precision()
        # Infere as dims dos embeddings do CACHE (librosa 320 / wav2vec2 768) em vez de
        # confiar no hardcode da config — assim o modelo casa com o Parquet existente.
        d_a, d_b, d_tab = self._dims_from_loader(train_data)
        if d_a and d_b and hasattr(self.model, "dim_a"):
            self.model.dim_a, self.model.dim_b = d_a, d_b
            log.info(f"Dims inferidas do cache: dim_a={d_a}, dim_b={d_b}")
        # dim_tab: na cross-attention só importa com o ramo tabular ligado (use_tabular);
        # modelos sem esse flag (GNNs: tabular no nó "video") sempre o consomem.
        if d_tab and hasattr(self.model, "dim_tab") and getattr(self.model, "use_tabular", True):
            self.model.dim_tab = d_tab
            log.info(f"Ramo tabular: dim_tab={d_tab} (inferido do cache)")
        # Modelos orientados a colunas (moe_fusion): dimensões de cada coluna usada.
        if hasattr(self.model, "infer_dims"):
            self.model.infer_dims(train_data.dataset)
        self._apply_pos_weight(train_data)
        self._lit_module = self._build_lit_module()
        if hasattr(self.model, "init_weights"):  # ex.: ramo inicializado de outro run
            self.model.init_weights(self._lit_module)
        self._init_weights_from_checkpoint()
        self._trainer = self._build_trainer()

        log.info(f"=== Treino de '{self._model_name()}' (Lightning) ===")
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
            threshold, _ = self._calibrate(val_ids, val_proba, val_labels, agg)
        else:
            threshold = float(thr_setting)
            log.info(f"Limiar fixo da config: {threshold:.3f}")
        self.threshold_ = threshold
        self.results["val"] = self._evaluate(val_ids, val_proba, val_labels)
        self.results["threshold"] = threshold
        return self.results

    def _calibrate(self, val_ids, val_proba, val_labels, agg) -> tuple[float, float]:
        """Calibra o limiar na val conforme o grupo ``aggregation`` (método/seleção)."""
        return calibrate_threshold(
            val_proba=val_proba,
            val_video_ids=val_ids,
            val_video_labels=val_labels,
            method="identity",
            metric=self._cfg_block("metrics").get("primary", "macro_f1"),
            selection=agg.get("calibration", "base_rate"),
            smooth_window=float(agg.get("smooth_window", 0.10)),
            target_pos_rate=_opt_float(agg.get("target_pos_rate")),
        )

    def recalibrate_on_val(self, val_loader) -> float:
        """Recalibra o limiar na val a partir do checkpoint carregado (SEM re-treinar).

        Chave do fluxo evaluate/submit: o limiar salvo no ``trainer_state.json`` fica
        congelado no checkpoint; mudar a estratégia de calibração só afetaria um re-treino.
        Este método roda inferência na val e recalcula ``self.threshold_`` com a config
        ``aggregation`` ATUAL — então ``aggregation.recalibrate=true`` faz a correção valer
        imediatamente em qualquer checkpoint existente.
        """
        ids, proba = self._infer(val_loader)
        labels = self._labels_from_loader(val_loader)
        thr, score = self._calibrate(ids, proba, labels, self._cfg_block("aggregation"))
        self.threshold_ = thr
        log.info(f"Recalibrado na val (sem re-treino): thr={thr:.3f} -> macro_f1={score:.4f}")
        return thr

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

    def predict_scores(self, data) -> dict[str, float]:
        """Score contínuo (sigmoid) por vídeo: ``{video_id: p1}``. Usado p/ submissão com
        probabilidades (formato ``video_id,p0,p1,pred``). O ``EnsembleTrainer`` herda isto
        e devolve a MÉDIA das probas dos membros (via seu ``_infer`` sobrescrito)."""
        ids, proba = self._infer(data)
        return {str(v): float(p) for v, p in zip(ids, proba, strict=False)}

    def video_outputs(self, data) -> dict[str, np.ndarray]:
        """Arrays a nível de vídeo p/ plots/relatórios (FASE 5): ids, y_true, y_proba, y_pred
        + o que o modelo expuser no ``predict_step`` (ex.: ``embedding`` pré-logit)."""
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit()/load() antes.")
        out = self.predict_outputs(data)
        ids, proba = out.pop("video_ids"), out.pop("proba")
        labels = self._labels_from_loader(data)
        sel = np.array([i for i in range(len(ids)) if str(ids[i]) in labels], dtype=np.int64)
        y_true = np.array([labels[str(ids[i])] for i in sel], dtype=np.int64)
        y_proba = proba[sel].astype(np.float32)
        y_pred = (y_proba >= self.threshold_).astype(np.int64)
        return {
            "video_ids": np.asarray(ids)[sel],
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
            **{key: value[sel] for key, value in out.items()},
        }

    def save(self, out_dir) -> None:
        """Salva o caminho do checkpoint + limiar (o peso fica no ckpt do Lightning).

        O ``trainer_state.json`` é autodescritivo: além do limiar e das dims, grava o
        ``model_name``, a arquitetura (``model_cfg``), o bloco ``model`` completo
        (``model_config``) e o Parquet de treino — o run dir pode então ser avaliado ou
        entrar num ensemble heterogêneo sem repetir o preset do treino.
        """
        import json

        from src.models.checkpoint_compat import snapshot_model_cfg

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    "threshold": self.threshold_,
                    "ckpt_path": self._ckpt_path,
                    "dim_a": int(getattr(self.model, "dim_a", 0) or 0),  # dims treinadas
                    "dim_b": int(getattr(self.model, "dim_b", 0) or 0),
                    "dim_tab": int(getattr(self.model, "dim_tab", 0) or 0),
                    "use_tabular": bool(getattr(self.model, "use_tabular", False)),
                    "pool": str(getattr(self.model, "pool", "mean")),
                    "tab_fusion": str(getattr(self.model, "tab_fusion", "late")),
                    "model_name": self._model_name(),
                    "model_cfg": snapshot_model_cfg(self.model),
                    # bloco `model` COMPLETO do treino (inclui sub-blocos como face/loss):
                    # load_trainer recria o modelo idêntico mesmo sem o preset na CLI.
                    "model_config": self._model_config(),
                    "parquet_path": self._parquet_path(),
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
        from src.models.checkpoint_compat import resolve_ckpt_path, resolve_model_cfg_for_load

        state = json.loads((Path(out_dir) / "trainer_state.json").read_text())
        trainer = cls(model=model, config=config)
        trainer.threshold_ = state.get("threshold")
        trainer.output_dir = str(out_dir)
        # ckpt_path do estado; se sumiu (run dir movido), procura o .ckpt DENTRO do
        # próprio run dir → checkpoint auto-contido e portátil.
        ckpt_path = resolve_ckpt_path(out_dir) or state.get("ckpt_path")
        trainer._ckpt_path = ckpt_path
        # Restaura a arquitetura treinada (hidden_channels, heads, … do model_cfg salvo;
        # runs antigos sem model_cfg: inferida dos shapes do state_dict) p/ casar c/ o ckpt.
        resolve_model_cfg_for_load(model, state, ckpt_path)
        # Restaura as dims treinadas p/ o módulo casar com os pesos do ckpt, e monta
        # um L.Trainer leve (sem logger/callbacks) p/ inferência (evaluate/predict).
        if state.get("dim_a") and state.get("dim_b") and hasattr(model, "dim_a"):
            model.dim_a, model.dim_b = int(state["dim_a"]), int(state["dim_b"])
        # Restaura o ramo tabular exatamente como no treino (senão o state_dict não casa).
        if "use_tabular" in state and hasattr(model, "use_tabular"):
            model.use_tabular = bool(state["use_tabular"])
        if state.get("dim_tab") and hasattr(model, "dim_tab"):
            model.dim_tab = int(state["dim_tab"])
        # Restaura pooling/fusão tabular da cross-attention (default = legado, p/
        # checkpoints antigos sem estes campos). Modelos sem esses atributos não são tocados.
        if hasattr(model, "pool"):
            model.pool = str(state.get("pool", "mean"))
        if hasattr(model, "tab_fusion"):
            model.tab_fusion = str(state.get("tab_fusion", "late"))
        trainer._lit_module = trainer._build_lit_module()
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
        out = self.predict_outputs(loader, embeddings=False)
        return out["video_ids"], out["proba"].astype(np.float32)

    def predict_outputs(self, loader, embeddings: bool = True) -> dict[str, np.ndarray]:
        """Tudo o que o ``predict_step`` devolve, concatenado por vídeo: ``video_ids``,
        ``proba`` e o que o modelo expuser (``router_weights``/``gates`` do moe_fusion).

        ``embeddings=True`` garante o vetor pré-logit (``embedding``, insumo do MoERouter)
        para QUALQUER modelo: se o ``predict_step`` não o devolve (cross-attention, GNNs),
        ele é capturado na entrada da camada ``Linear(·, 1)`` que gera o logit.
        """
        capture = _PrelogitCapture(self._lit_module) if embeddings else None
        try:
            outputs = self._trainer.predict(
                self._lit_module, dataloaders=loader, ckpt_path=self._ckpt_path
            )
        finally:
            if capture is not None:
                capture.remove()
        parts: dict[str, list] = {}
        for out in outputs:
            for key, value in out.items():
                if key == "video_ids":
                    parts.setdefault(key, []).extend(list(value))
                else:
                    parts.setdefault(key, []).append(value.detach().float().cpu().numpy())
        result = {
            key: np.asarray(value) if key == "video_ids" else np.concatenate(value, axis=0)
            for key, value in parts.items()
        }
        if capture is not None and "embedding" not in result:
            emb = capture.embedding_for(result["proba"])
            if emb is not None:
                result["embedding"] = emb
        return result

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

    def _model_name(self) -> str:
        try:
            return str(self.config.model.name)
        except Exception:  # noqa: BLE001
            return type(self.model).__name__

    def _model_config(self) -> dict[str, Any] | None:
        try:
            from omegaconf import OmegaConf

            model = self.config.model
            return (
                OmegaConf.to_container(model, resolve=True)
                if OmegaConf.is_config(model)
                else dict(model)
            )
        except Exception:  # noqa: BLE001
            return None

    def _parquet_path(self) -> str | None:
        try:
            return str(self.config.data.paths.parquet_path)
        except Exception:  # noqa: BLE001
            return None

    def _init_weights_from_checkpoint(self) -> None:
        """Fine-tune: carrega pesos de ``checkpoint=<run_dir|.ckpt>`` no ``mode=train``.

        Só o ``state_dict`` (sem otimizador), ``strict=False`` — chaves ausentes/extras
        são logadas e ignoradas. Em evaluate/submit ``checkpoint`` continua sendo o run
        dir a avaliar (lido pelo ``main``); aqui só vale para o treino.
        """
        ckpt_arg = getattr(self.config, "checkpoint", None)
        if not ckpt_arg:
            return
        from src.models.checkpoint_compat import load_state_dict_from_ckpt, resolve_ckpt_path

        ckpt_path = resolve_ckpt_path(ckpt_arg)
        if not ckpt_path:
            log.warning(f"Fine-tune: checkpoint não encontrado em {ckpt_arg}")
            return

        state_dict = load_state_dict_from_ckpt(ckpt_path)
        missing, unexpected = self._lit_module.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(f"Fine-tune: {len(missing)} chaves ausentes no ckpt (ignoradas).")
        if unexpected:
            log.warning(f"Fine-tune: {len(unexpected)} chaves extras no ckpt (ignoradas).")
        log.info(f"Fine-tune: pesos carregados de {ckpt_path}")

    def _apply_pos_weight(self, train_loader) -> None:
        """Resolve ``model.pos_weight`` (``auto`` = neg/pos do treino) p/ modelos que o usam."""
        if not hasattr(self.model, "pos_weight"):
            return
        from src.models.lightning_utils import resolve_pos_weight

        pw = resolve_pos_weight(getattr(self.model, "pos_weight", None), train_loader)
        if hasattr(self.model, "_resolved_pos_weight"):
            self.model._resolved_pos_weight = pw

    def _evaluate(self, ids, proba, labels) -> dict[str, Any]:
        preds = aggregate_to_video(proba, ids, method="identity", threshold=self.threshold_)
        scores = {str(v): float(p) for v, p in zip(ids, proba, strict=False)}
        return evaluate_video_predictions(video_labels=labels, video_pred=preds, video_score=scores)


class _PrelogitCapture:
    """Captura a ENTRADA da camada ``Linear(·, 1)`` que gera o logit (o vetor pré-logit) de
    qualquer ``LightningModule``, sem mudar o código do modelo.

    Registra hooks em toda ``Linear`` com 1 saída; no fim fica com a camada cujo
    ``sigmoid(saída)`` reproduz a ``proba`` do ``predict_step`` — cabeças auxiliares e
    escores de atenção (por janela) não batem e são descartados.
    """

    def __init__(self, module: Any) -> None:
        from torch import nn

        self._records: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        self._handles = [
            layer.register_forward_hook(self._hook(name))
            for name, layer in module.named_modules()
            if isinstance(layer, nn.Linear) and layer.out_features == 1
        ]

    def _hook(self, name: str):
        inputs, logits = self._records.setdefault(name, ([], []))

        def hook(_layer, args, output) -> None:
            if output.dim() == 2 and args and args[0].dim() == 2:  # (B, d) → (B, 1)
                inputs.append(args[0].detach().float().cpu().numpy())
                logits.append(output.detach().float().cpu().numpy()[:, 0])

        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

    def embedding_for(self, proba: np.ndarray) -> np.ndarray | None:
        """Entradas da (última) camada cujo ``sigmoid`` reproduz ``proba``; ``None`` se nenhuma."""
        from scipy.special import expit

        for inputs, logits in reversed(list(self._records.values())):
            if logits:
                z = np.concatenate(logits)
                if z.shape == proba.shape and np.allclose(expit(z), proba, atol=1e-4):
                    return np.concatenate(inputs)
        log.info("Vetor pré-logit não identificado: nenhuma Linear(·, 1) reproduz a proba.")
        return None


def _opt_float(value: Any) -> float | None:
    """``None``/``"null"``/``"none"`` → ``None``; senão ``float`` (tolera string da CLI)."""
    if value is None or (isinstance(value, str) and value.lower() in ("null", "none", "")):
        return None
    return float(value)


def _set_cuda_matmul_precision() -> None:
    """TF32 nas matmuls em CUDA (ganho de velocidade sem efeito em CPU/MPS)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
    except Exception:  # noqa: BLE001
        pass
