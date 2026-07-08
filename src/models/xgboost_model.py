"""XGBoostModel — classificador de janela baseado em XGBoost (família sklearn).

Embrulha ``XGBClassifier`` sob o contrato ``BaseModel``. Roda em CPU (ou GPU se
``tree_method="gpu_hist"``). Consome a mesma matriz achatada do RandomForest:
``X = [audio_emb ‖ text_emb ‖ tabular]`` por janela. A agregação janela→vídeo
e a calibração de limiar ficam no ``SklearnTrainer``.

Por que XGBoost aqui?
- Gradient boosting tende a superar RF em features mistas (embeddings densos + tabular).
- ``scale_pos_weight`` lida nativamente com o desbalanceamento (~17% positivo).
- Suporta importâncias de feature (gain, weight, cover) para análise de interpretabilidade.

``scale_pos_weight="auto"`` (default): calculado como ``n_neg / n_pos`` no ``fit``,
que é a regra recomendada pelo XGBoost para dados desbalanceados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.base.model import BaseModel
from src.logger import get_logger
from src.models.registry import register_model

log = get_logger("models.xgboost")


@register_model("xgboost", family="sklearn")
class XGBoostModel(BaseModel):
    """XGBoost binário (A/H sim/não) a nível de janela.

    Attributes:
        clf: ``XGBClassifier`` subjacente.
        feature_names: nomes das colunas de X (importâncias legíveis).
        classes_: classes vistas no ``fit`` (espera-se ``[0, 1]``).
        scale_pos_weight_mode: ``"auto"`` (calculado no fit) ou float fixo.
    """

    def __init__(
        self,
        clf,
        feature_names: list[str] | None = None,
        scale_pos_weight_mode: str | float = "auto",
    ) -> None:
        self.clf = clf
        self.feature_names = feature_names
        self.classes_: np.ndarray | None = None
        self.scale_pos_weight_mode = scale_pos_weight_mode

    # ==========================================================================
    # Treino e inferência (contrato §6.4)
    # ==========================================================================

    def fit(self, X, y, sample_weight=None) -> XGBoostModel:
        """Treina o XGBoost nas janelas de treino.

        Se ``scale_pos_weight_mode="auto"``, calcula ``n_neg / n_pos`` e injeta no
        estimador antes do fit (sem re-criar o objeto — preserva outros hiperparâmetros).

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
        n_neg = int((y == 0).sum())
        log.info(f"Treinando XGBoost: X={X.shape}, positivos={n_pos}/{len(y)}")

        if self.scale_pos_weight_mode == "auto" and n_pos > 0:
            spw = n_neg / n_pos
            log.info(f"scale_pos_weight automático: {spw:.2f} ({n_neg} neg / {n_pos} pos)")
            self.clf.set_params(scale_pos_weight=spw)

        self.clf.fit(X, y, sample_weight=sample_weight)
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
        """Importâncias de feature por ganho (gain) — mais interpretável que weight."""
        if self.classes_ is None:
            return None
        try:
            imp = self.clf.get_booster().get_score(importance_type="gain")
            # get_score devolve dict esparso {f0: v, f3: v, ...}; alinha ao vetor completo
            n_features = self.clf.n_features_in_
            arr = np.zeros(n_features, dtype=np.float32)
            for k, v in imp.items():
                idx = int(k[1:]) if k.startswith("f") else int(k)
                if idx < n_features:
                    arr[idx] = float(v)
            # Normaliza p/ comparação com RF (soma=1)
            total = arr.sum()
            return (arr / total) if total > 0 else arr
        except Exception:
            # Fallback: feature_importances_ (weight)
            fi = getattr(self.clf, "feature_importances_", None)
            return np.asarray(fi, dtype=np.float32) if fi is not None else None

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
                "scale_pos_weight_mode": self.scale_pos_weight_mode,
                "model_type": "xgboost",
            },
            path,
        )
        log.info(f"XGBoostModel salvo em: {path}")

    @classmethod
    def load(cls, path) -> XGBoostModel:
        """Carrega o modelo de um arquivo joblib."""
        payload = joblib.load(Path(path))
        model = cls(
            clf=payload["clf"],
            feature_names=payload.get("feature_names"),
            scale_pos_weight_mode=payload.get("scale_pos_weight_mode", "auto"),
        )
        model.classes_ = payload.get("classes_")
        return model

    # ==========================================================================
    # Factory
    # ==========================================================================

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> XGBoostModel:
        """Cria o modelo a partir de ``configs/model/xgboost.yaml``.

        Chaves suportadas: ``n_estimators``, ``max_depth``, ``learning_rate``,
        ``subsample``, ``colsample_bytree``, ``min_child_weight``, ``gamma``,
        ``reg_alpha``, ``reg_lambda``, ``scale_pos_weight`` ("auto" ou float),
        ``n_jobs``, ``random_state``, ``tree_method``, ``eval_metric``.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("XGBoost não instalado. Execute: uv add xgboost") from e

        spw_raw = config.get("scale_pos_weight", "auto")
        spw_mode = "auto" if str(spw_raw).lower() == "auto" else float(spw_raw)
        spw_init = 1.0 if spw_mode == "auto" else float(spw_mode)

        clf = XGBClassifier(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.05),
            subsample=config.get("subsample", 0.8),
            colsample_bytree=config.get("colsample_bytree", 0.8),
            min_child_weight=config.get("min_child_weight", 3),
            gamma=config.get("gamma", 0.0),
            reg_alpha=config.get("reg_alpha", 0.0),
            reg_lambda=config.get("reg_lambda", 1.0),
            scale_pos_weight=spw_init,
            n_jobs=config.get("n_jobs", -1),
            random_state=config.get("random_state", 42),
            tree_method=config.get("tree_method", "hist"),
            eval_metric=config.get("eval_metric", "logloss"),
            verbosity=0,
        )
        return cls(
            clf=clf,
            feature_names=config.get("feature_names"),
            scale_pos_weight_mode=spw_mode,
        )
