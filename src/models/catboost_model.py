"""CatBoostModel — classificador de janela baseado em CatBoost (família sklearn)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.base.model import BaseModel
from src.logger import get_logger
from src.models.registry import register_model

log = get_logger("models.catboost")


@register_model("catboost", family="sklearn")
class CatBoostModel(BaseModel):
    """CatBoost binário (A/H sim/não) a nível de janela."""

    def __init__(self, clf, feature_names: list[str] | None = None) -> None:
        self.clf = clf
        self.feature_names = feature_names
        self.classes_: np.ndarray | None = None

    def fit(self, X, y, sample_weight=None) -> CatBoostModel:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        n_pos = int((y == 1).sum())
        log.info(f"Treinando CatBoost: X={X.shape}, positivos={n_pos}/{len(y)}")
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.clf.fit(X, y, **fit_kwargs)
        self.classes_ = np.array(self.clf.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)].astype(np.float32)
        return np.full(X.shape[0], 0.0, dtype=np.float32)

    def feature_importances(self) -> np.ndarray | None:
        if self.classes_ is None:
            return None
        try:
            fi = self.clf.get_feature_importance()
            arr = np.asarray(fi, dtype=np.float32)
            total = arr.sum()
            return (arr / total) if total > 0 else arr
        except Exception:
            return None

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self.clf,
                "feature_names": self.feature_names,
                "classes_": self.classes_,
                "model_type": "catboost",
            },
            path,
        )
        log.info(f"CatBoostModel salvo em: {path}")

    @classmethod
    def load(cls, path) -> CatBoostModel:
        payload = joblib.load(Path(path))
        model = cls(clf=payload["clf"], feature_names=payload.get("feature_names"))
        model.classes_ = payload.get("classes_")
        return model

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CatBoostModel:
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError("CatBoost não instalado. Execute: uv add catboost") from e

        clf = CatBoostClassifier(
            iterations=int(config.get("n_estimators", 300)),
            depth=int(config.get("depth", 6)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            l2_leaf_reg=float(config.get("l2_leaf_reg", 3.0)),
            auto_class_weights=config.get("auto_class_weights", "Balanced"),
            random_seed=int(config.get("random_state", 42)),
            thread_count=int(config.get("thread_count", -1)),
            verbose=int(config.get("verbose", 0)),
            allow_writing_files=False,
        )
        return cls(clf=clf, feature_names=config.get("feature_names"))
