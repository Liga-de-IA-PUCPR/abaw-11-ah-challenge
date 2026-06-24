"""RandomForestModel — classificador de janela default do BAH (família sklearn).

Embrulha ``RandomForestClassifier`` sob o contrato ``BaseModel`` (README §6.4).
Roda 100% em CPU — **não importa torch/lightning**. Consome a matriz achatada
``X = [audio_emb ‖ text_emb ‖ tabular]`` por janela (README §6.2) e devolve a
probabilidade da classe positiva (A/H = 1) por janela; a agregação janela→vídeo +
limiar calibrado fica no ``SklearnTrainer``.

Robusto ao ruído de rótulo (Fleiss' κ de frame ≈ 0,41) e ao desbalanceamento
(~17% das janelas são A/H) via ``class_weight``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.base.model import BaseModel
from src.logger import get_logger
from src.models.registry import register_model

log = get_logger("models.random_forest")


@register_model("random_forest", family="sklearn")
class RandomForestModel(BaseModel):
    """RandomForest binário (A/H sim/não) a nível de janela.

    Attributes:
        clf: ``RandomForestClassifier`` subjacente.
        feature_names: nomes das colunas de X (importâncias legíveis).
        classes_: classes vistas no ``fit`` (espera-se ``[0, 1]``).
    """

    def __init__(self, clf: RandomForestClassifier, feature_names: list[str] | None = None) -> None:
        """Inicializa o modelo com um estimador sklearn já configurado.

        Args:
            clf: ``RandomForestClassifier`` (ainda não treinado).
            feature_names: nomes das features alinhados às colunas de X (opcional).
        """
        self.clf = clf
        self.feature_names = feature_names
        self.classes_: np.ndarray | None = None

    # ==========================================================================
    # Treino e inferência (contrato §6.4)
    # ==========================================================================

    def fit(self, X, y, sample_weight=None) -> RandomForestModel:
        """Treina o RandomForest nas janelas de treino.

        Args:
            X: matriz de features (n_windows, d) — ``[audio_emb ‖ text_emb ‖ tabular]``.
            y: rótulos de janela em {0, 1}.
            sample_weight: pesos opcionais por amostra.

        Returns:
            self (treinado).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        log.info(f"Treinando RandomForest: X={X.shape}, positivos={int(y.sum())}/{len(y)}")
        self.clf.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probabilidade da classe positiva (A/H = 1) por janela.

        Devolve **apenas a coluna da classe 1** (contrato §6.4), localizada pela
        posição de ``1`` em ``classes_`` para robustez.

        Returns:
            Array (n,) de probabilidades em [0, 1].
        """
        X = np.asarray(X, dtype=np.float32)
        proba = self.clf.predict_proba(X)
        return proba[:, self._positive_column()].astype(np.float32)

    def feature_importances(self) -> np.ndarray | None:
        """Importâncias de Gini por feature (alinhadas a ``feature_names``)."""
        if self.classes_ is None:
            return None
        return np.asarray(self.clf.feature_importances_, dtype=np.float32)

    # ==========================================================================
    # Persistência (joblib) — CPU, sem torch
    # ==========================================================================

    def save(self, path) -> None:
        """Serializa o modelo + metadados de feature com joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self.clf,
                "feature_names": self.feature_names,
                "classes_": self.classes_,
                "model_type": "random_forest",
            },
            path,
        )
        log.info(f"Modelo salvo em: {path}")

    @classmethod
    def load(cls, path) -> RandomForestModel:
        """Carrega o modelo de um arquivo joblib."""
        payload = joblib.load(Path(path))
        model = cls(clf=payload["clf"], feature_names=payload.get("feature_names"))
        model.classes_ = payload.get("classes_")
        return model

    # ==========================================================================
    # Factory
    # ==========================================================================

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RandomForestModel:
        """Cria o modelo a partir de ``configs/model/random_forest.yaml``.

        Chaves (README §7, grupo ``model``): ``n_estimators``, ``max_depth``,
        ``min_samples_leaf``, ``max_features``, ``class_weight``, ``n_jobs``,
        ``random_state``. ``feature_names`` é opcional (injetado na FASE 3).
        """
        clf = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 600),
            max_depth=config.get("max_depth"),
            min_samples_leaf=config.get("min_samples_leaf", 2),
            max_features=config.get("max_features", "sqrt"),
            class_weight=config.get("class_weight", "balanced_subsample"),
            n_jobs=config.get("n_jobs", -1),
            random_state=config.get("random_state", 42),
        )
        return cls(clf=clf, feature_names=config.get("feature_names"))

    def _positive_column(self) -> int:
        """Índice da coluna correspondente à classe 1 em ``predict_proba``."""
        if self.classes_ is None:
            return 1
        idx = np.where(self.classes_ == 1)[0]
        return int(idx[0]) if len(idx) else 1
