import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.audio = df["audio_emb"].to_numpy()  # (N, 768)
        self.text = df["text_emb"].to_numpy()  # (N, 384)
        self.labels = df["label"].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "audio_emb": torch.tensor(self.audio[idx], dtype=torch.float32),
            "text_emb": torch.tensor(self.text[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
