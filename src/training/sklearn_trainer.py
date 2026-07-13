"""SklearnTrainer — treino do modelo de janela + agregação janela→vídeo (CPU).

Contrato ``BaseTrainer`` (README §6.4): ``fit`` → ``evaluate`` → ``predict`` →
``save`` / ``load``. **Não importa torch**: matriz de janelas (FASE 3) -> modelo
sklearn -> probas de janela -> agrega janela→vídeo -> calibração opcional ->
limiar calibrado na val -> métricas sklearn a nível de vídeo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.base.model import BaseModel
from src.base.trainer import BaseTrainer
from src.logger import get_logger
from src.training.aggregation import (
    calibrate_threshold,
    calibrate_threshold_from_video_scores,
    video_scores,
)
from src.training.metrics import evaluate_video_predictions
from src.training.score_calibration import (
    apply_calibrator,
    fit_temperature,
    serialize_calibrator,
)

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
    """Trainer para modelos de janela sklearn + agregação janela→vídeo (CPU)."""

    family: str = "sklearn"

    def __init__(self, model: BaseModel, config: Any) -> None:
        super().__init__(model=model, cfg=config)
        self.config = config
        agg = self._cfg_block("aggregation")
        self.method: str = agg.get("method", "mean_proba")
        self._thr_setting = agg.get("threshold", "auto")
        self._score_calibration: str = str(agg.get("score_calibration", "none"))
        self._calibration: str = agg.get("calibration", "smooth")
        self._smooth_window: float = float(agg.get("smooth_window", 0.10))
        target = agg.get("target_pos_rate")
        self._target_pos_rate: float | None = None if target in (None, "null") else float(target)
        self.threshold_: float | None = None
        self._calibrator: dict[str, Any] | None = None
        self.results: dict[str, Any] = {}

    def _cfg_block(self, name: str) -> dict[str, Any]:
        block = getattr(self.config, name, None)
        if block is None and isinstance(self.config, dict):
            block = self.config.get(name, {})
        if block is None:
            return {}
        return dict(block) if not isinstance(block, dict) else block

    def fit(self, train_data: FeatureSplit, val_data: FeatureSplit) -> dict[str, Any]:
        if train_data.y is None:
            raise ValueError("Split de treino sem rótulos (y is None).")

        log.info("=== Treino do modelo de janela (sklearn, CPU) ===")
        self.model.fit(train_data.X, train_data.y, sample_weight=None)
        n_pos = int(np.asarray(train_data.y).sum())
        self.results["train"] = {"n_windows": int(len(train_data.y)), "n_positives": n_pos}

        log.info("=== Calibração (score + limiar) + avaliação na validação ===")
        val_proba = self.model.predict_proba(val_data.X)
        raw_scores = video_scores(val_proba, val_data.video_ids, self.method)
        val_ids = [v for v in raw_scores if v in val_data.video_labels]
        y_true = np.array([val_data.video_labels[v] for v in val_ids], dtype=np.int64)
        s_raw = np.array([raw_scores[v] for v in val_ids], dtype=np.float32)

        if self._score_calibration == "temperature":
            temperature = fit_temperature(s_raw, y_true)
            self._calibrator = serialize_calibrator("temperature", T=temperature)
        else:
            self._calibrator = None

        s_cal = apply_calibrator(s_raw, self._calibrator)

        if self._thr_setting == "auto":
            if self._calibrator is not None:
                threshold, _ = calibrate_threshold_from_video_scores(
                    s_cal,
                    y_true,
                    metric=self._cfg_block("metrics").get("primary", "macro_f1"),
                    selection=self._calibration,
                    smooth_window=self._smooth_window,
                    target_pos_rate=self._target_pos_rate,
                )
            else:
                threshold, _ = calibrate_threshold(
                    val_proba=val_proba,
                    val_video_ids=val_data.video_ids,
                    val_video_labels=val_data.video_labels,
                    method=self.method,
                    metric=self._cfg_block("metrics").get("primary", "macro_f1"),
                    selection=self._calibration,
                    smooth_window=self._smooth_window,
                    target_pos_rate=self._target_pos_rate,
                )
        else:
            threshold = float(self._thr_setting)
            log.info(f"Limiar fixo da config: {threshold:.3f}")
        self.threshold_ = threshold

        self.results["val"] = self._evaluate_split(val_data, val_proba)
        self.results["threshold"] = self.threshold_
        self.results["method"] = self.method
        self.results["score_calibration"] = self._score_calibration
        self.results["calibrator"] = self._calibrator
        self.results["feature_importances"] = self._top_feature_importances(k=20)
        return self.results

    def evaluate(self, data: FeatureSplit) -> dict[str, Any]:
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit() antes de evaluate().")
        proba = self.model.predict_proba(data.X)
        return self._evaluate_split(data, proba)

    def predict(self, data: FeatureSplit) -> dict[str, int]:
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit() antes de predict().")
        proba = self.model.predict_proba(data.X)
        scores = self._calibrated_scores_dict(video_scores(proba, data.video_ids, self.method))
        return {vid: int(score >= self.threshold_) for vid, score in scores.items()}

    def video_outputs(self, data: FeatureSplit) -> dict[str, np.ndarray]:
        if self.threshold_ is None:
            raise RuntimeError("Limiar não calibrado: chame fit()/load() antes.")
        window_proba = self.model.predict_proba(data.X)
        scores = self._calibrated_scores_dict(
            video_scores(window_proba, data.video_ids, self.method)
        )
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

    def save(self, out_dir) -> None:
        import json

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(out_dir / "model.joblib")
        (out_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    "threshold": self.threshold_,
                    "method": self.method,
                    "score_calibration": self._score_calibration,
                    "calibrator": self._calibrator,
                },
                indent=2,
            )
        )
        log.info(f"SklearnTrainer salvo em: {out_dir}")

    @classmethod
    def load(cls, out_dir, model: BaseModel, config: Any) -> SklearnTrainer:
        import json

        out_dir = Path(out_dir)
        state = json.loads((out_dir / "trainer_state.json").read_text())
        trainer = cls(model=model, config=config)
        trainer.threshold_ = state.get("threshold")
        trainer.method = state.get("method", trainer.method)
        trainer._score_calibration = state.get("score_calibration", trainer._score_calibration)
        trainer._calibrator = state.get("calibrator")
        return trainer

    def _calibrated_scores_dict(self, raw: dict[str, float]) -> dict[str, float]:
        if not self._calibrator:
            return raw
        ids = list(raw.keys())
        arr = np.array([raw[v] for v in ids], dtype=np.float32)
        cal = apply_calibrator(arr, self._calibrator)
        return {v: float(cal[i]) for i, v in enumerate(ids)}

    def _evaluate_split(self, split: FeatureSplit, window_proba: np.ndarray) -> dict[str, Any]:
        scores = self._calibrated_scores_dict(
            video_scores(window_proba, split.video_ids, self.method)
        )
        preds = {
            vid: int(score >= self.threshold_)
            for vid, score in scores.items()
            if vid in split.video_labels
        }
        return evaluate_video_predictions(
            video_labels=split.video_labels, video_pred=preds, video_score=scores
        )

    def _top_feature_importances(self, k: int = 20) -> list[tuple[str, float]]:
        imp = self.model.feature_importances()
        if imp is None:
            return []
        names = getattr(self.model, "feature_names", None)
        if names is None or len(names) != len(imp):
            names = [f"f{i}" for i in range(len(imp))]
        order = np.argsort(imp)[::-1][:k]
        return [(str(names[i]), float(imp[i])) for i in order]
