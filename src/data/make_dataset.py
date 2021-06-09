# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any, Tuple

import torch
from torchvision import datasets

data_dir = Path(__file__).parents[2] / "data"


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


def mnist():

    # Download and load the training data
    train = MNIST(data_dir, download=True, train=True)
    # Download and load the test data
    test = MNIST(data_dir, download=True, train=False)

    return train, test
