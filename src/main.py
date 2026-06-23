import lightning as L
import polars as pl
import torch

from modeling.data import EmbeddingDataset
from modeling.model_cross_attention import CrossAttention, CrossAttentionFusionModel

parquet_path = "../data/processed/text_audio.parquet"

df = pl.read_parquet(parquet_path).with_columns(
    [
        pl.col("audio_emb").cast(pl.Array(pl.Float32, 768)),
        pl.col("text_emb").cast(pl.Array(pl.Float32, 384)),
    ]
)

dataset = EmbeddingDataset(df)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)


sample = df.sample()

model = CrossAttentionFusionModel(
    dim_a=len(sample["audio_emb"][0]), dim_b=len(sample["text_emb"][0])
)
module = CrossAttention(model)

trainer = L.Trainer(limit_train_batches=100, max_epochs=100)
trainer.fit(model=module, train_dataloaders=train_loader)
