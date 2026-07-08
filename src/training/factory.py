"""Factory de trainer: escolhe o trainer pela família do modelo.

``create_trainer(family, model, cfg)``:
  - "sklearn"   -> ``SklearnTrainer``   (CPU, sem torch).
  - "lightning" -> ``LightningTrainer`` (import LAZY de lightning).

``load_trainer(family, out_dir, cfg)`` reconstrói o modelo (``create_model`` /
``RandomForestModel.load``) e o trainer e delega ao ``BaseTrainer.load`` da
família (assinatura ``load(out_dir, model, config)``).

O import do ``LightningTrainer`` acontece DENTRO da função, só quando
``family == "lightning"`` — mantendo o caminho RF livre de torch/lightning.

> **Device:** os trainers resolvem o device **internamente** via
> ``resolve_device(cfg.device)`` (README §7 / regra device-Lightning-opcional);
> não há parâmetro ``device`` nas assinaturas — o ``cfg`` já o carrega.
"""

from __future__ import annotations

from typing import Any

from src.base.trainer import BaseTrainer
from src.logger import get_logger

log = get_logger("training.factory")


def create_trainer(family: str, model: Any, cfg: Any) -> BaseTrainer:
    """Cria o trainer adequado à família do modelo.

    Args:
        family: "sklearn" | "lightning" (devolvido por ``create_model``).
        model: objeto de modelo (``BaseModel`` sklearn ou ``CrossAttentionFusion``).
        cfg: configuração do experimento (Hydra/``DictConfig`` ou dataclass);
            o device é lido de ``cfg.device`` dentro do trainer.

    Returns:
        Instância de ``BaseTrainer``.

    Raises:
        ValueError: família desconhecida.
    """
    if family == "sklearn":
        from src.training.sklearn_trainer import SklearnTrainer

        log.info("Trainer: SklearnTrainer (CPU, sem lightning).")
        return SklearnTrainer(model=model, config=cfg)

    if family == "lightning":
        # IMPORT LAZY — só aqui torch/lightning entram em cena.
        from src.training.lightning_trainer import LightningTrainer

        log.info("Trainer: LightningTrainer (import lazy de lightning).")
        return LightningTrainer(model=model, config=cfg)

    raise ValueError(f"Família de trainer desconhecida: '{family}'")


def load_trainer(family: str, out_dir: Any, cfg: Any) -> BaseTrainer:
    """Recarrega um trainer treinado a partir de ``out_dir`` (checkpoint).

    Args:
        family: "sklearn" | "lightning".
        out_dir: diretório do checkpoint (escrito por ``trainer.save``).
        cfg: configuração do experimento.

    Returns:
        Instância de ``BaseTrainer`` pronta para ``evaluate``/``predict``.

    Raises:
        ValueError: família desconhecida.
    """
    from pathlib import Path

    out_dir = Path(out_dir)

    if family == "sklearn":
        import joblib as _jl

        from src.training.sklearn_trainer import SklearnTrainer

        model_path = out_dir / "model.joblib"
        # Peek no model_type salvo para despachar à classe correta.
        payload = _jl.load(model_path)
        model_type = payload.get("model_type", "random_forest")

        if model_type == "xgboost":
            from src.models.xgboost_model import XGBoostModel

            model = XGBoostModel.load(model_path)
        elif model_type == "lightgbm":
            from src.models.lightgbm_model import LightGBMModel

            model = LightGBMModel.load(model_path)
        elif model_type == "extra_trees":
            from src.models.extra_trees import ExtraTreesModel

            model = ExtraTreesModel.load(model_path)
        elif model_type == "logistic_regression":
            from src.models.logistic_regression import LogisticRegressionModel

            model = LogisticRegressionModel.load(model_path)
        elif model_type == "catboost":
            from src.models.catboost_model import CatBoostModel

            model = CatBoostModel.load(model_path)
        elif model_type == "mlp":
            from src.models.mlp_model import MLPModel

            model = MLPModel.load(model_path)
        elif model_type in ("stacking", "stacking_catboost", "stacking_cat_rf"):
            from src.models.stacking_model import StackingModel

            model = StackingModel.load(model_path)
        else:
            from src.models.random_forest import RandomForestModel

            model = RandomForestModel.load(model_path)

        log.info(f"Trainer: SklearnTrainer recarregado (model_type={model_type}).")
        return SklearnTrainer.load(out_dir, model=model, config=cfg)

    if family == "lightning":
        # IMPORT LAZY — só aqui torch/lightning entram em cena.
        from src.models.registry import create_model
        from src.training.lightning_trainer import LightningTrainer

        model, _ = create_model(cfg.model.name, cfg.model)
        log.info("Trainer: LightningTrainer recarregado (import lazy de lightning).")
        return LightningTrainer.load(out_dir, model=model, config=cfg)

    raise ValueError(f"Família de trainer desconhecida: '{family}'")
