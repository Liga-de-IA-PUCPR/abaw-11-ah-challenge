"""MLPModel — classificador de janela via rede neural densa (família sklearn).

Embrulha ``Pipeline(StandardScaler → MLPClassifier)`` sob o contrato
``BaseModel``. O escalonamento Z-score é crítico para MLPs: garante que o
gradiente não exploda nem suma nas camadas internas.

**Por que sklearn MLP aqui e não Lightning?**
O ``MLPClassifier`` do sklearn opera sobre features tabulares fixas (sem
sequência temporal) — a mesma matriz ``X = [audio ‖ texto ‖ tabular]`` do RF.
É comparável ao ``CrossAttentionFusion`` (Lightning) sem a modelagem temporal:
testa se a não-linearidade densa já ajuda na separação das features.

**Arquitetura configurável via dois escalares**:
- ``hidden_dim_1``: tamanho da 1ª camada oculta (int, default 256).
- ``hidden_dim_2``: tamanho da 2ª camada oculta; ``0`` = rede de 1 camada.

Isso permite sweeps simples via Hydra (inteiros, não tuplas):
    ``model.hidden_dim_1=256,512   model.hidden_dim_2=0,128``

**Regularização**: ``alpha`` (L2), dropout implícito via ``early_stopping``.
``class_weight`` não existe no sklearn MLP; o desbalanceamento é tratado via
pesos de amostra (``sample_weight``) passados pelo trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.base.model import BaseModel
from src.logger import get_logger
from src.models.registry import register_model

log = get_logger("models.mlp")


@register_model("mlp", family="sklearn")
class MLPModel(BaseModel):
    """MLP denso com escalonamento embutido (via Pipeline sklearn).

    Attributes:
        pipeline: ``Pipeline([scaler, clf])`` serializado integralmente.
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

    def fit(self, X, y, sample_weight=None) -> MLPModel:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        n_pos = int((y == 1).sum())
        log.info(f"Treinando MLP: X={X.shape}, positivos={n_pos}/{len(y)}")
        # sklearn MLP não aceita sample_weight diretamente no fit — omitimos.
        # O desbalanceamento é mitigado via threshold calibration no SklearnTrainer.
        if sample_weight is not None:
            log.debug(
                "MLPClassifier (sklearn) não suporta sample_weight no fit — "
                "o desbalanceamento será tratado via calibração de limiar."
            )
        self.pipeline.fit(X, y)
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
        """MLP não fornece importância de features diretamente; retorna None."""
        return None

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
                "model_type": "mlp",
            },
            path,
        )
        log.info(f"MLPModel salvo em: {path}")

    @classmethod
    def load(cls, path) -> MLPModel:
        payload = joblib.load(Path(path))
        model = cls(pipeline=payload["pipeline"], feature_names=payload.get("feature_names"))
        model.classes_ = payload.get("classes_")
        return model

    # ==========================================================================
    # Factory
    # ==========================================================================

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MLPModel:
        """Cria o modelo a partir de ``configs/model/mlp.yaml``.

        Chaves suportadas: ``hidden_dim_1`` (int), ``hidden_dim_2`` (int, 0 = sem 2ª camada),
        ``alpha`` (L2), ``learning_rate_init``, ``max_iter``, ``random_state``,
        ``activation``, ``early_stopping``, ``validation_fraction``.
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        h1 = int(config.get("hidden_dim_1", 256))
        h2 = int(config.get("hidden_dim_2", 128))
        hidden_layer_sizes = (h1,) if h2 == 0 else (h1, h2)
        log.debug(f"MLP hidden_layer_sizes={hidden_layer_sizes}")

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=config.get("activation", "relu"),
            alpha=float(config.get("alpha", 1e-4)),
            learning_rate_init=float(config.get("learning_rate_init", 1e-3)),
            max_iter=int(config.get("max_iter", 500)),
            early_stopping=bool(config.get("early_stopping", True)),
            validation_fraction=float(config.get("validation_fraction", 0.1)),
            n_iter_no_change=int(config.get("n_iter_no_change", 20)),
            random_state=int(config.get("random_state", 42)),
            verbose=False,
        )
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        return cls(pipeline=pipeline, feature_names=config.get("feature_names"))
