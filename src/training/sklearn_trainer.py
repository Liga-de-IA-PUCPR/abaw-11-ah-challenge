"""SklearnTrainer — treino do modelo de janela + agregação janela→vídeo (CPU).

Contrato ``BaseTrainer`` (README §6.4): ``fit`` → ``evaluate`` → ``predict`` →
``save`` / ``load``. **Não importa torch**: matriz de janelas (FASE 3) -> RF ->
probas de janela -> agrega janela→vídeo -> limiar calibrado na val -> métricas
sklearn a nível de vídeo.

Os splits chegam como ``FeatureSplit`` (visão do cache Parquet §6.2), passados aos
métodos (o ``__init__`` segue ``BaseTrainer(model, cfg)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.base.model import BaseModel
from src.base.trainer import BaseTrainer
from src.logger import get_logger
from src.training.aggregation import aggregate_to_video, calibrate_threshold, video_scores
from src.training.metrics import evaluate_video_predictions

log = get_logger("training.sklearn_trainer")


@dataclass
class FeatureSplit:
    """Visão de um split em memória (achatada do cache Parquet, §6.2)."""

    X: np.ndarray  # (n_windows, d) = [audio_emb ‖ text_emb ‖ tabular]
    y: np.ndarray | None  # (n_windows,) em {0,1}; None no test
    video_ids: np.ndarray  # (n_windows,) id do vídeo por janela
    participant_ids: np.ndarray  # (n_windows,) participante por janela
    video_labels: dict[str, int]  # global_ah por vídeo (vazio no test puro)


class SklearnTrainer(BaseTrainer):
    """Trainer para modelos de janela sklearn + agregação janela→vídeo (CPU).

    Attributes:
        model: ``BaseModel`` de janela (ex.: ``RandomForestModel``).
        config: config com grupos ``trainer`` (sklearn), ``aggregation``, ``metrics``.
        method: método de agregação (``cfg.aggregation.method``).
        threshold_: limiar calibrado por ``fit`` na validação.
    """

    family: str = "sklearn"

    def __init__(self, model: BaseModel, config: Any) -> None:
        super().__init__(model=model, cfg=config)
        # Atalho legível para o ``cfg`` (mantém o corpo dos métodos claro).
        self.config = config
        agg = self._cfg_block("aggregation")
        self.method: str = agg.get("method", "mean_proba")
        self._thr_setting = agg.get("threshold", "auto")
        self.threshold_: float | None = None
        self.results: dict[str, Any] = {}

    def _cfg_block(self, name: str) -> dict[str, Any]:
        """Lê um bloco da config tolerando dataclass/``DictConfig``/``dict``."""
        block = getattr(self.config, name, None)
        if block is None and isinstance(self.config, dict):
            block = self.config.get(name, {})
        if block is None:
            return {}
        return dict(block) if not isinstance(block, dict) else block

    # ==========================================================================
    # Contrato BaseTrainer (README §6.4)
    # ==========================================================================

    def fit(self, train_data: FeatureSplit, val_data: FeatureSplit) -> dict[str, Any]:
        """Treina o modelo de janela e calibra o limiar na validação.

        sklearn não tem épocas: um único ``model.fit`` (class_weight via config do
        modelo, §7). Depois calibra o limiar de agregação que maximiza o Macro-F1.

        Returns:
            Dict com métricas de janela do treino + métricas de vídeo da validação.
        """
        if train_data.y is None:
            raise ValueError("Split de treino sem rótulos (y is None).")

        log.info("=== Treino do modelo de janela (sklearn, CPU) ===")
        self.model.fit(train_data.X, train_data.y, sample_weight=None)
        n_pos = int(np.asarray(train_data.y).sum())
        self.results["train"] = {"n_windows": int(len(train_data.y)), "n_positives": n_pos}

        log.info("=== Calibração do limiar + avaliação na validação ===")
        val_proba = self.model.predict_proba(val_data.X)
        if self._thr_setting == "auto":
            threshold, _ = calibrate_threshold(
                val_proba=val_proba,
                val_video_ids=val_data.video_ids,
                val_video_labels=val_data.video_labels,
                method=self.method,
                metric=self._cfg_block("metrics").get("primary", "macro_f1"),
            )
        else:
            threshold = float(self._thr_setting)
            log.info(f"Limiar fixo da config: {threshold:.3f}")
        self.threshold_ = threshold

        self.results["val"] = self._evaluate_split(val_data, val_proba)
        self.results["threshold"] = self.threshold_
        self.results["method"] = self.method
        self.results["feature_importances"] = self._top_feature_importances(k=20)
        return self.results

    def evaluate(self, data: FeatureSplit) -> dict[str, Any]:
        """Avalia um split rotulado a nível de vídeo (no limiar calibrado)."""
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit() antes de evaluate().")
        proba = self.model.predict_proba(data.X)
        return self._evaluate_split(data, proba)

    def predict(self, data: FeatureSplit) -> dict[str, int]:
        """Predição A/H a nível de vídeo: ``{video_id: pred 0/1}`` (FASE 5/submissão)."""
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit() antes de predict().")
        proba = self.model.predict_proba(data.X)
        return aggregate_to_video(proba, data.video_ids, self.method, self.threshold_)

    def video_outputs(self, data: FeatureSplit) -> dict[str, np.ndarray]:
        """Arrays a nível de vídeo p/ plots/relatórios (FASE 5): ids, y_true, y_proba, y_pred.

        Alinha os scores agregados aos rótulos de vídeo conhecidos. Consumido por
        ``_write_eval_report`` (matriz de confusão, PR, curva de limiar).
        """
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit()/load() antes.")
        window_proba = self.model.predict_proba(data.X)
        scores = video_scores(window_proba, data.video_ids, self.method)
        ids = [v for v in scores if v in data.video_labels]
        y_true = np.array([data.video_labels[v] for v in ids], dtype=np.int64)
        y_proba = np.array([scores[v] for v in ids], dtype=np.float32)
        y_pred = (y_proba >= self.threshold_).astype(np.int64)
        return {
            "video_ids": np.asarray(ids),
            "y_true": y_true,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }

    # ==========================================================================
    # Persistência (joblib do modelo + limiar/método)
    # ==========================================================================

    def save(self, out_dir) -> None:
        """Salva o modelo (joblib) + estado do trainer (limiar/método)."""
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(out_dir / "model.joblib")
        (out_dir / "trainer_state.json").write_text(
            json.dumps({"threshold": self.threshold_, "method": self.method}, indent=2)
        )
        log.info(f"SklearnTrainer salvo em: {out_dir}")

    @classmethod
    def load(cls, out_dir, model: BaseModel, config: Any) -> SklearnTrainer:
        """Recarrega o trainer (espera ``model`` já carregado via ``BaseModel.load``)."""
        import json

        out_dir = Path(out_dir)
        state = json.loads((out_dir / "trainer_state.json").read_text())
        trainer = cls(model=model, config=config)
        trainer.threshold_ = state.get("threshold")
        trainer.method = state.get("method", trainer.method)
        return trainer

    # ==========================================================================
    # Internos
    # ==========================================================================

    def _evaluate_split(self, split: FeatureSplit, window_proba: np.ndarray) -> dict[str, Any]:
        """Agrega janelas→vídeo e calcula o relatório de métricas a nível de vídeo."""
        preds = aggregate_to_video(window_proba, split.video_ids, self.method, self.threshold_)
        scores = video_scores(window_proba, split.video_ids, self.method)
        return evaluate_video_predictions(
            video_labels=split.video_labels, video_pred=preds, video_score=scores
        )

    def _top_feature_importances(self, k: int = 20) -> list[tuple[str, float]]:
        """Top-k features por importância (vazio se o modelo não expõe)."""
        imp = self.model.feature_importances()
        if imp is None:
            return []
        names = getattr(self.model, "feature_names", None)
        if names is None or len(names) != len(imp):
            names = [f"f{i}" for i in range(len(imp))]
        order = np.argsort(imp)[::-1][:k]
        return [(str(names[i]), float(imp[i])) for i in order]
