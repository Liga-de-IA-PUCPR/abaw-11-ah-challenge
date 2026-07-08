"""LightGBMModel — classificador de janela baseado em LightGBM (família sklearn).

Embrulha ``LGBMClassifier`` sob o contrato ``BaseModel``. LightGBM é tipicamente
mais rápido que XGBoost em CPU e frequentemente comparável em acurácia. Interface
idêntica ao ``RandomForestModel`` e ``XGBoostModel`` — o ``SklearnTrainer`` os
trata de forma uniforme.

Vantagens para este problema:
- Velocidade: treina 5-10× mais rápido que XGBoost em features de alta dimensão
  (embeddings de 768d tornam o treino mais lento para RF/XGB).
- ``class_weight="balanced"`` nativo, sem precisar calcular ``scale_pos_weight``.
- Importâncias de feature via ganho, split count ou SHAP (gain é o default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.base.model import BaseModel
from src.logger import get_logger
from src.models.registry import register_model

log = get_logger("models.lightgbm")


@register_model("lightgbm", family="sklearn")
class LightGBMModel(BaseModel):
    """LightGBM binário (A/H sim/não) a nível de janela.

    Attributes:
        clf: ``LGBMClassifier`` subjacente.
        feature_names: nomes das colunas de X (importâncias legíveis).
        classes_: classes vistas no ``fit`` (espera-se ``[0, 1]``).
    """

    def __init__(self, clf, feature_names: list[str] | None = None) -> None:
        self.clf = clf
        self.feature_names = feature_names
        self.classes_: np.ndarray | None = None

    # ==========================================================================
    # Treino e inferência (contrato §6.4)
    # ==========================================================================

    def fit(self, X, y, sample_weight=None) -> LightGBMModel:
        """Treina o LightGBM nas janelas de treino.

        Args:
            X: matriz de features (n_windows, d).
            y: rótulos de janela em {0, 1}.
            sample_weight: pesos opcionais por amostra.

        Returns:
            self (treinado).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()

        n_pos = int((y == 1).sum())
        log.info(f"Treinando LightGBM: X={X.shape}, positivos={n_pos}/{len(y)}")

        # feature_name=feature_names expõe os nomes nos relatórios do LightGBM.
        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
        if self.feature_names and len(self.feature_names) == X.shape[1]:
            fit_kwargs["feature_name"] = self.feature_names

        self.clf.fit(X, y, **fit_kwargs)
        self.classes_ = np.array(self.clf.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidade da classe positiva (A/H = 1) por janela.

        Returns:
            Array (n,) de probabilidades em [0, 1].
        """
        X = np.asarray(X, dtype=np.float32)
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)].astype(np.float32)
        return np.full(X.shape[0], 1.0 if classes == [1] else 0.0, dtype=np.float32)

    def feature_importances(self) -> np.ndarray | None:
        """Importâncias de feature por ganho (gain) — normalizado p/ soma=1."""
        if self.classes_ is None:
            return None
        fi = getattr(self.clf, "feature_importances_", None)
        if fi is None:
            return None
        arr = np.asarray(fi, dtype=np.float32)
        total = arr.sum()
        return (arr / total) if total > 0 else arr

    # ==========================================================================
    # Persistência (joblib)
    # ==========================================================================

    def save(self, path) -> None:
        """Serializa o modelo + metadados com joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self.clf,
                "feature_names": self.feature_names,
                "classes_": self.classes_,
                "model_type": "lightgbm",
            },
            path,
        )
        log.info(f"LightGBMModel salvo em: {path}")

    @classmethod
    def load(cls, path) -> LightGBMModel:
        """Carrega o modelo de um arquivo joblib."""
        payload = joblib.load(Path(path))
        model = cls(clf=payload["clf"], feature_names=payload.get("feature_names"))
        model.classes_ = payload.get("classes_")
        return model

    # ==========================================================================
    # Factory
    # ==========================================================================

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LightGBMModel:
        """Cria o modelo a partir de ``configs/model/lightgbm.yaml``.

        Chaves suportadas: ``n_estimators``, ``num_leaves``, ``max_depth``,
        ``learning_rate``, ``subsample``, ``colsample_bytree``,
        ``min_child_samples``, ``reg_alpha``, ``reg_lambda``,
        ``class_weight``, ``n_jobs``, ``random_state``, ``verbose``.
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError("LightGBM não instalado. Execute: uv add lightgbm") from e

        clf = LGBMClassifier(
            n_estimators=config.get("n_estimators", 300),
            num_leaves=config.get("num_leaves", 63),
            max_depth=config.get("max_depth", -1),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            subsample_freq=config.get("subsample_freq", 1),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            min_child_samples=config.get("min_child_samples", 20),
            reg_alpha=config.get("reg_alpha", 0.0),
            reg_lambda=config.get("reg_lambda", 0.0),
            class_weight=config.get("class_weight", "balanced"),
            n_jobs=config.get("n_jobs", -1),
            random_state=config.get("random_state", 42),
            verbose=config.get("verbose", -1),
        )
        return cls(clf=clf, feature_names=config.get("feature_names"))
