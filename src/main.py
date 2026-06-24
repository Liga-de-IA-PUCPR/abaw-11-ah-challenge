import hydra
import lightning as L
import polars as pl
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
from torch.utils.data import random_split

from modeling.data import EmbeddingDataset
from modeling.model_cross_attention import CrossAttention, CrossAttentionFusionModel


def load_data(cfg: DictConfig):
    df = pl.read_parquet(cfg.parquet_path).with_columns(
        [
            pl.col("audio_emb").cast(pl.Array(pl.Float32, cfg.dim_a)),
            pl.col("text_emb").cast(pl.Array(pl.Float32, cfg.dim_b)),
        ]
    )

    dataset = EmbeddingDataset(df)

    train_size = int(cfg.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)

    return train_loader, val_loader


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train_loader, val_loader = load_data(
        DictConfig({**cfg.data, "dim_a": cfg.model.dim_a, "dim_b": cfg.model.dim_b})
    )

    model = CrossAttentionFusionModel(
        dim_a=cfg.model.dim_a,
        dim_b=cfg.model.dim_b,
        common_dim=cfg.model.common_dim,
        num_heads=cfg.model.num_heads,
        num_classes=cfg.model.num_classes,
    )
    module = CrossAttention(model, lr=cfg.model.lr)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri="sqlite:///mlflow.db",
        log_model=True,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=mlf_logger,
    )
    trainer.fit(
        model=module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


if __name__ == "__main__":
    main()
