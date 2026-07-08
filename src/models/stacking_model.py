"""StackingModel — meta-ensemble RF + XGBoost + LightGBM (família sklearn).

Embrulha ``StackingClassifier`` do sklearn sob o contrato ``BaseModel``.

**Arquitetura do stacking**:
1. **Estimadores base** (camada 1):
   - ``RandomForestClassifier`` — captura interações não-lineares globais.
   - ``XGBClassifier`` — gradient boosting com scale_pos_weight.
   - ``LGBMClassifier`` — gradient boosting rápido, excelente em features densas.
   - ``CatBoostClassifier`` — opcional, ativado com ``use_catboost: true``.
2. **Meta-classificador** (camada 2):
   - ``LogisticRegression`` — aprende a combinar as probabilidades base.
   - Recebe previsões OOF (out-of-fold) dos estimadores base durante o treino,
     o que **previne overfitting** do meta-modelo nas mesmas amostras de treino.

**Por que stacking funciona aqui?**
Cada base model comete erros diferentes: RF alavanca interações globais, XGBoost
foca nos resíduos mais difíceis, LGBM é mais rápido com a mesma qualidade. O
meta-modelo aprende quais bases confiar em quais situações.

O ``StackingClassifier`` do sklearn usa ``cv=5`` folds para gerar previsões OOF
— isso aumenta o tempo de treino em ~5×, mas produz um meta-modelo mais generalizável.

Os hiperparâmetros de cada base model são totalmente configuráveis via YAML
(prefixos ``base_rf_*``, ``base_xgb_*``, ``base_lgbm_*``, ``cat_*``), permitindo
usar os melhores valores encontrados nos sweeps individuais.

Instalação de dependências: ``uv add xgboost lightgbm``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.base.model import BaseModel
from src.logger import get_logger
from src.models.registry import register_model

log = get_logger("models.stacking")


@register_model("stacking", family="sklearn")
class StackingModel(BaseModel):
    """Meta-ensemble RF + XGBoost + LightGBM → LogisticRegression.

    Attributes:
        clf: ``StackingClassifier`` subjacente com base models e meta-modelo.
        feature_names: nomes das colunas de X.
        classes_: classes vistas no ``fit``.
    """

    def __init__(
        self, clf, feature_names: list[str] | None = None, model_type: str = "stacking"
    ) -> None:
        self.clf = clf
        self.feature_names = feature_names
        self.classes_: np.ndarray | None = None
        self._model_type = model_type

    # ==========================================================================
    # Treino e inferência
    # ==========================================================================

    def fit(self, X, y, sample_weight=None) -> StackingModel:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        log.info(
            f"Treinando Stacking: X={X.shape}, pos={n_pos}/{len(y)} "
            f"(cv={self.clf.cv} folds — aguarde ~{self.clf.cv}× tempo de treino)"
        )
        # Propaga scale_pos_weight para XGBoost antes do fit (se estiver nos estimators).
        if n_pos > 0:
            spw = n_neg / n_pos
            for name, est in self.clf.estimators:
                if hasattr(est, "set_params") and hasattr(est, "scale_pos_weight"):
                    est.set_params(scale_pos_weight=spw)
                    log.debug(f"  [{name}] scale_pos_weight={spw:.2f}")

        # StackingClassifier não propaga sample_weight a todos os base estimators —
        # exigiria suporte em RF + XGB + LGBM simultaneamente. O desbalanceamento
        # é tratado via class_weight/scale_pos_weight nos base models e via
        # calibração de limiar no SklearnTrainer.
        if sample_weight is not None:
            log.debug(
                "StackingModel.fit: sample_weight ignorado — StackingClassifier não "
                "aceita sample_weight propagado a todos os base estimators."
            )
        self.clf.fit(X, y)
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
        """Stacking não tem importância direta; retorna None."""
        return None

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
                "model_type": self._model_type,
            },
            path,
        )
        log.info(f"StackingModel salvo em: {path}")

    @classmethod
    def load(cls, path) -> StackingModel:
        payload = joblib.load(Path(path))
        model = cls(
            clf=payload["clf"],
            feature_names=payload.get("feature_names"),
            model_type=payload.get("model_type", "stacking"),
        )
        model.classes_ = payload.get("classes_")
        return model

    # ==========================================================================
    # Factory
    # ==========================================================================

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> StackingModel:
        """Cria o modelo a partir de ``configs/model/stacking.yaml``.

        Chaves suportadas — RF base:
        - ``base_rf_n_estimators``   : árvores (default 200).
        - ``base_rf_max_depth``      : profundidade máxima; None = irrestrito (default None).
        - ``base_rf_min_samples_leaf``: mínimo de amostras por folha (default 1).
        - ``base_rf_max_features``   : "sqrt" | "log2" | float (default "sqrt").

        Chaves suportadas — XGBoost base:
        - ``base_xgb_n_estimators``  : iterações (default 100).
        - ``base_xgb_max_depth``     : profundidade máxima (default 6).
        - ``base_xgb_learning_rate`` : taxa de aprendizado (default 0.05).
        - ``base_xgb_subsample``     : fração de amostras por árvore (default 0.8).
        - ``base_xgb_colsample_bytree``: fração de colunas por árvore (default 0.8).
        - ``base_xgb_min_child_weight``: peso mínimo de filho (default 1).
        - ``base_xgb_gamma``         : regularização mínima de ganho (default 0.0).
        - ``base_xgb_reg_alpha``     : L1 (default 0.0).
        - ``base_xgb_reg_lambda``    : L2 (default 1.0).

        Chaves suportadas — LightGBM base:
        - ``base_lgbm_n_estimators`` : iterações (default 100).
        - ``base_lgbm_num_leaves``   : folhas por árvore (default 31).
        - ``base_lgbm_max_depth``    : -1 = irrestrito (default -1).
        - ``base_lgbm_learning_rate``: taxa de aprendizado (default 0.05).
        - ``base_lgbm_subsample``    : fração de amostras (default 0.8).
        - ``base_lgbm_subsample_freq``: frequência de bagging (default 1).
        - ``base_lgbm_colsample_bytree``: fração de colunas (default 0.8).
        - ``base_lgbm_min_child_samples``: mínimo de amostras por folha (default 20).
        - ``base_lgbm_reg_alpha``    : L1 (default 0.0).
        - ``base_lgbm_reg_lambda``   : L2 (default 0.0).

        Chaves suportadas — meta e geral:
        - ``cv``          : folds para previsões OOF (default 5).
        - ``meta_C``      : regularização do meta LogisticRegression (default 1.0).
        - ``n_jobs``      : paralelismo (default -1).
        - ``random_state``: seed (default 42).
        """
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rs = int(config.get("random_state", 42))
        n_jobs = int(config.get("n_jobs", -1))

        # --- RF base ---
        _rf_max_depth = config.get("base_rf_max_depth")
        rf = RandomForestClassifier(
            n_estimators=int(config.get("base_rf_n_estimators", 200)),
            max_depth=int(_rf_max_depth) if _rf_max_depth is not None else None,
            min_samples_leaf=int(config.get("base_rf_min_samples_leaf", 1)),
            max_features=config.get("base_rf_max_features", "sqrt"),
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=rs,
        )

        # --- XGBoost base (opcional — desativado por use_xgb=false na config) ---
        use_xgb_flag = bool(config.get("use_xgb", True))
        try:
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
                n_estimators=int(config.get("base_xgb_n_estimators", 100)),
                max_depth=int(config.get("base_xgb_max_depth", 6)),
                learning_rate=float(config.get("base_xgb_learning_rate", 0.05)),
                subsample=float(config.get("base_xgb_subsample", 0.8)),
                colsample_bytree=float(config.get("base_xgb_colsample_bytree", 0.8)),
                min_child_weight=int(config.get("base_xgb_min_child_weight", 1)),
                gamma=float(config.get("base_xgb_gamma", 0.0)),
                reg_alpha=float(config.get("base_xgb_reg_alpha", 0.0)),
                reg_lambda=float(config.get("base_xgb_reg_lambda", 1.0)),
                scale_pos_weight=1.0,  # ajustado dinamicamente no fit
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=n_jobs,
                random_state=rs,
                verbosity=0,
            )
            has_xgb = use_xgb_flag
        except ImportError:
            log.warning("XGBoost não instalado — omitido do stacking.")
            has_xgb = False
        if not use_xgb_flag:
            has_xgb = False
            log.info("XGBoost desativado por use_xgb=false na config.")

        # --- LightGBM base (opcional — desativado por use_lgbm=false na config) ---
        use_lgbm_flag = bool(config.get("use_lgbm", True))
        try:
            from lightgbm import LGBMClassifier

            lgbm = LGBMClassifier(
                n_estimators=int(config.get("base_lgbm_n_estimators", 100)),
                num_leaves=int(config.get("base_lgbm_num_leaves", 31)),
                max_depth=int(config.get("base_lgbm_max_depth", -1)),
                learning_rate=float(config.get("base_lgbm_learning_rate", 0.05)),
                subsample=float(config.get("base_lgbm_subsample", 0.8)),
                subsample_freq=int(config.get("base_lgbm_subsample_freq", 1)),
                colsample_bytree=float(config.get("base_lgbm_colsample_bytree", 0.8)),
                min_child_samples=int(config.get("base_lgbm_min_child_samples", 20)),
                reg_alpha=float(config.get("base_lgbm_reg_alpha", 0.0)),
                reg_lambda=float(config.get("base_lgbm_reg_lambda", 0.0)),
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=rs,
                verbose=-1,
            )
            has_lgbm = use_lgbm_flag
        except ImportError:
            log.warning("LightGBM não instalado — omitido do stacking.")
            has_lgbm = False
        if not use_lgbm_flag:
            has_lgbm = False
            log.info("LightGBM desativado por use_lgbm=false na config.")

        estimators: list[tuple[str, Any]] = [("rf", rf)]
        if has_xgb:
            estimators.append(("xgb", xgb))
        if has_lgbm:
            estimators.append(("lgbm", lgbm))

        # --- CatBoost (opcional — ativado por use_catboost=true na config) ---
        use_catboost = bool(config.get("use_catboost", False))
        if use_catboost:
            try:
                from catboost import CatBoostClassifier  # lazy: só aqui

                cat = CatBoostClassifier(
                    iterations=int(config.get("cat_n_estimators", 100)),
                    depth=int(config.get("cat_depth", 8)),
                    learning_rate=float(config.get("cat_learning_rate", 0.1)),
                    l2_leaf_reg=float(config.get("cat_l2_leaf_reg", 5.0)),
                    auto_class_weights="Balanced",
                    verbose=False,
                    random_seed=rs,
                )
                estimators.append(("catboost", cat))
                log.info("CatBoost (melhores hiperparâmetros do sweep) adicionado ao stacking.")
            except ImportError:
                log.warning("CatBoost não instalado — omitido do stacking_catboost.")
                use_catboost = False

        if len(estimators) < 2:
            log.warning(
                "Apenas 1 estimador base disponível no stacking — "
                "instale xgboost e lightgbm para resultados melhores."
            )

        # --- Meta-classificador (com scaler para normalizar as probabilidades OOF) ---
        meta = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=float(config.get("meta_C", 1.0)),
                        class_weight="balanced",
                        max_iter=2000,
                        solver="saga",
                        random_state=rs,
                    ),
                ),
            ]
        )

        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=int(config.get("cv", 5)),
            stack_method="predict_proba",
            n_jobs=1,  # n_jobs=-1 no StackingClassifier pode causar conflito com XGB
            passthrough=False,
        )
        if use_catboost and not has_xgb and not has_lgbm:
            _model_type = "stacking_cat_rf"
        elif use_catboost:
            _model_type = "stacking_catboost"
        else:
            _model_type = "stacking"
        return cls(clf=clf, feature_names=config.get("feature_names"), model_type=_model_type)


# Registros adicionais — mesma classe, configs diferentes.
register_model("stacking_catboost", family="sklearn")(StackingModel)
register_model("stacking_cat_rf", family="sklearn")(StackingModel)
