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
        Instância de ``BaseTrainer`` (``SklearnTrainer`` ou ``LightningTrainer``).

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

    Reconstrói o modelo da família (RF via ``RandomForestModel.load``;
    cross-attention via ``create_model``) e delega ao ``BaseTrainer.load`` da
    família, cuja assinatura é ``load(out_dir, model, config)``.

    Args:
        family: "sklearn" | "lightning".
        out_dir: diretório do checkpoint (escrito por ``trainer.save``).
        cfg: configuração do experimento; o device é lido de ``cfg.device``
            dentro do trainer (sem parâmetro ``device`` explícito).

    Returns:
        Instância de ``BaseTrainer`` pronta para ``evaluate``/``predict``.

    Raises:
        ValueError: família desconhecida.
    """
    from pathlib import Path

    out_dir = Path(out_dir)

    if family == "sklearn":
        from src.models.random_forest import RandomForestModel
        from src.training.sklearn_trainer import SklearnTrainer

        model = RandomForestModel.load(out_dir / "model.joblib")
        log.info("Trainer: SklearnTrainer recarregado (CPU, sem lightning).")
        return SklearnTrainer.load(out_dir, model=model, config=cfg)

    if family == "lightning":
        # IMPORT LAZY — só aqui torch/lightning entram em cena.
        from src.models.registry import create_model
        from src.training.lightning_trainer import LightningTrainer

        model, _ = create_model(cfg.model.name, cfg.model)
        log.info("Trainer: LightningTrainer recarregado (import lazy de lightning).")
        return LightningTrainer.load(out_dir, model=model, config=cfg)

    raise ValueError(f"Família de trainer desconhecida: '{family}'")
