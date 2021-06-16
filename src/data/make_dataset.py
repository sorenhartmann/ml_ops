# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Tuple

import torch
from torchvision import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class MNIST(datasets.MNIST):
    @property
    def raw_folder(self) -> str:
        return self.root / "raw" / "MNIST"

    @property
    def processed_folder(self) -> str:
        return self.root / "processed" / "MNIST"

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = (
            self.data.unsqueeze(-3).to(torch.float) / 255
        )  # view as (bs, 1, 28, 28)


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './', batch_size=32):

        super().__init__()

        self.data_dir = Path(data_dir)
        
        self.batch_size = batch_size

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):

        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False)


    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)

