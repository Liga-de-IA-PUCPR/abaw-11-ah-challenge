"""LogisticRegressionModel — classificador de janela linear (família sklearn).

Embrulha ``Pipeline(StandardScaler → LogisticRegression)`` sob o contrato
``BaseModel``.

**Por que escalonar?**
Os embeddings de texto (RoBERTa, 768-d) e áudio (LibROSA, ~260-d) têm escalas
muito distintas. O StandardScaler garante que o otimizador L-BFGS convirja
rápido e que a regularização L2 (parâmetro ``C``) incida uniformemente sobre
todas as dimensões.

**Custo computacional baixo** — útil como baseline linear forte antes de
ensemble mais pesados. Com embeddings de qualidade (RoBERTa + wav2vec2), LR
frequentemente supera RF em problemas onde a separação é aproximadamente linear
no espaço latente.

**Hiperparâmetro crítico**: ``C`` (inverso da regularização). Valores pequenos
(0.001–0.1) regularizam fortemente; valores grandes (10–100) deixam o modelo
mais livre — na presença de embeddings pré-treinados, tende a precisar de
regularização moderada (C ≈ 0.1–1.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.base.model import BaseModel
from src.logger import get_logger
from src.models.registry import register_model

log = get_logger("models.logistic_regression")


@register_model("logistic_regression", family="sklearn")
class LogisticRegressionModel(BaseModel):
    """Regressão Logística com normalização Z-score embutida.

    O pipeline interno (``scaler → clf``) é salvo/carregado como um único
    objeto joblib, preservando os parâmetros do scaler ajustados no treino.

    Attributes:
        pipeline: ``sklearn.pipeline.Pipeline`` (scaler + LogisticRegression).
        feature_names: nomes das colunas de X.
        classes_: classes vistas no ``fit``.
    """

    def __init__(self, pipeline, feature_names: list[str] | None = None) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.classes_: np.ndarray | None = None

    # ==========================================================================
    # Treino e inferência
    # ==========================================================================

    def fit(self, X, y, sample_weight=None) -> LogisticRegressionModel:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        n_pos = int((y == 1).sum())
        log.info(f"Treinando LogisticRegression: X={X.shape}, positivos={n_pos}/{len(y)}")
        fit_params = {}
        if sample_weight is not None:
            fit_params["clf__sample_weight"] = sample_weight
        self.pipeline.fit(X, y, **fit_params)
        self.classes_ = np.array(self.pipeline.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        proba = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)].astype(np.float32)
        return np.full(X.shape[0], 0.0, dtype=np.float32)

    def feature_importances(self) -> np.ndarray | None:
        """Coeficientes (valor absoluto) como proxy de importância."""
        clf = self.pipeline.named_steps.get("clf")
        if clf is None or not hasattr(clf, "coef_"):
            return None
        coef = np.abs(clf.coef_).ravel()
        total = coef.sum()
        return (coef / total).astype(np.float32) if total > 0 else coef.astype(np.float32)

    # ==========================================================================
    # Persistência
    # ==========================================================================

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "classes_": self.classes_,
                "model_type": "logistic_regression",
            },
            path,
        )
        log.info(f"LogisticRegressionModel salvo em: {path}")

    @classmethod
    def load(cls, path) -> LogisticRegressionModel:
        payload = joblib.load(Path(path))
        model = cls(pipeline=payload["pipeline"], feature_names=payload.get("feature_names"))
        model.classes_ = payload.get("classes_")
        return model

    # ==========================================================================
    # Factory
    # ==========================================================================

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LogisticRegressionModel:
        """Cria o modelo a partir de ``configs/model/logistic_regression.yaml``.

        Chaves suportadas: ``C``, ``penalty``, ``solver``, ``max_iter``,
        ``class_weight``, ``random_state``, ``n_jobs``, ``tol``.

        O solver ``saga`` suporta L1 e L2 e é o mais eficiente para features
        de alta dimensão (~1000-d). ``lbfgs`` é mais estável para L2 puro.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        solver = config.get("solver", "saga")

        clf = LogisticRegression(
            C=float(config.get("C", 1.0)),
            solver=solver,
            max_iter=int(config.get("max_iter", 5000)),
            class_weight=config.get("class_weight", "balanced"),
            random_state=config.get("random_state", 42),
            tol=float(config.get("tol", 1e-4)),
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        return cls(pipeline=pipeline, feature_names=config.get("feature_names"))
