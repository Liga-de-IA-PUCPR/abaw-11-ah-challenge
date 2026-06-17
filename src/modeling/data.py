import lightning as L
from torch.utils.data import DataLoader, Dataset


class BAHDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load a single sample
        sample = load_your_data(self.file_paths[index])  # e.g. image, tensor, etc.
        label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


class BAHDatamodule(L.LightningDataModule):
    def __init__(self, train_files, val_files, test_files, batch_size=32):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Called on every GPU — create dataset splits here
        if stage == "fit":
            self.train_ds = MyDataset(self.train_files, ...)
            self.val_ds   = MyDataset(self.val_files, ...)
        if stage == "test":
            self.test_ds  = MyDataset(self.test_files, ...)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)