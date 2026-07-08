"""ExtraTreesModel — classificador de janela baseado em ExtraTrees (família sklearn).

Embrulha ``ExtraTreesClassifier`` sob o contrato ``BaseModel``. ExtraTrees difere
do RandomForest em **dois pontos-chave**:

1. **Splits aleatórios**: em vez de buscar o melhor ponto de corte entre os
   candidatos, ExtraTrees sorteia o limiar aleatoriamente — reduz variância e
   acelera o treino às custas de um bias ligeiramente maior.
2. **Uso de toda a amostra de treino** (sem bootstrap por default): cada árvore
   vê todos os exemplos, o que reduz variância adicional.

Isso o torna especialmente útil em features com alta dimensionalidade (768-d de
embeddings de texto + áudio), onde a busca exaustiva do melhor split do RF pode
sofrer de overfitting.

``class_weight="balanced"`` lida com o desbalanceamento (~17% positivo) sem
precisar de oversampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.base.model import BaseModel
from src.logger import get_logger
from src.models.registry import register_model

log = get_logger("models.extra_trees")


@register_model("extra_trees", family="sklearn")
class ExtraTreesModel(BaseModel):
    """ExtraTrees binário (A/H sim/não) a nível de janela.

    Attributes:
        clf: ``ExtraTreesClassifier`` subjacente.
        feature_names: nomes das colunas de X (importâncias legíveis).
        classes_: classes vistas no ``fit`` (espera-se ``[0, 1]``).
    """

    def __init__(self, clf, feature_names: list[str] | None = None) -> None:
        self.clf = clf
        self.feature_names = feature_names
        self.classes_: np.ndarray | None = None

    # ==========================================================================
    # Treino e inferência
    # ==========================================================================

    def fit(self, X, y, sample_weight=None) -> ExtraTreesModel:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        n_pos = int((y == 1).sum())
        log.info(f"Treinando ExtraTrees: X={X.shape}, positivos={n_pos}/{len(y)}")
        self.clf.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.array(self.clf.classes_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)].astype(np.float32)
        return np.full(X.shape[0], 1.0 if classes == [1] else 0.0, dtype=np.float32)

    def feature_importances(self) -> np.ndarray | None:
        fi = getattr(self.clf, "feature_importances_", None)
        return np.asarray(fi, dtype=np.float32) if fi is not None else None

    # ==========================================================================
    # Persistência
    # ==========================================================================

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self.clf,
                "feature_names": self.feature_names,
                "classes_": self.classes_,
                "model_type": "extra_trees",
            },
            path,
        )
        log.info(f"ExtraTreesModel salvo em: {path}")

    @classmethod
    def load(cls, path) -> ExtraTreesModel:
        payload = joblib.load(Path(path))
        model = cls(clf=payload["clf"], feature_names=payload.get("feature_names"))
        model.classes_ = payload.get("classes_")
        return model

    # ==========================================================================
    # Factory
    # ==========================================================================

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExtraTreesModel:
        """Cria o modelo a partir de ``configs/model/extra_trees.yaml``.

        Chaves suportadas: ``n_estimators``, ``max_depth``, ``min_samples_leaf``,
        ``max_features``, ``class_weight``, ``n_jobs``, ``random_state``,
        ``bootstrap`` (False por default — comportamento ExtraTrees canônico).
        """
        from sklearn.ensemble import ExtraTreesClassifier

        clf = ExtraTreesClassifier(
            n_estimators=config.get("n_estimators", 400),
            max_depth=config.get("max_depth"),
            min_samples_leaf=config.get("min_samples_leaf", 2),
            max_features=config.get("max_features", "sqrt"),
            class_weight=config.get("class_weight", "balanced"),
            bootstrap=config.get("bootstrap", False),
            n_jobs=config.get("n_jobs", -1),
            random_state=config.get("random_state", 42),
        )
        return cls(clf=clf, feature_names=config.get("feature_names"))
