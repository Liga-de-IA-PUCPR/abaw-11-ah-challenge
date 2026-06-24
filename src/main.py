from pathlib import Path

import hydra
import lightning as L
import polars as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from modeling.data import EmbeddingDataset
from modeling.model_cross_attention import CrossAttention, CrossAttentionFusionModel


def load_data(cfg: DictConfig):
    df = pl.read_parquet(cfg.parquet_path).with_columns(
        [
            pl.col("audio_emb").cast(pl.Array(pl.Float32, cfg.dim_a)),
            pl.col("text_emb").cast(pl.Array(pl.Float32, cfg.dim_b)),
        ]
    )

    def _ids_from_split(name: str) -> list[str]:
        path = Path(cfg.split_dir) / f"{name}.txt"
        return [line.split(",", 1)[0] for line in path.read_text().strip().splitlines()]

    train_ids = _ids_from_split("train")
    val_ids = _ids_from_split("val")
    test_ids = _ids_from_split("test")

    train_df = df.filter(pl.col("id").is_in(train_ids))
    val_df = df.filter(pl.col("id").is_in(val_ids))
    test_df = df.filter(pl.col("id").is_in(test_ids))

    train_ds = EmbeddingDataset(train_df)
    val_ds = EmbeddingDataset(val_df)
    test_ds = EmbeddingDataset(test_df)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size)

    return train_loader, val_loader, test_loader


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    train_loader, val_loader, test_loader = load_data(
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

    wandb_logger = WandbLogger(
        project=cfg.experiment_name,
        log_model=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch:02d}-{val_loss:.4f}"
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=20), checkpoint_callback],
    )
    trainer.fit(
        model=module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
