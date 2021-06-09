import pytest
import torch

from src.data.make_dataset import mnist


@pytest.fixture()
def datasets():
    train_set, test_set = mnist()
    return train_set, test_set


def test_lengths(datasets):

    train_set, test_set = datasets

    assert len(train_set) == 60_000
    assert len(test_set) == 10_000


def test_train_test(datasets):

    train_set, test_set = datasets

    assert not torch.equal(test_set.data, train_set.data)
    assert not torch.equal(test_set.targets, train_set.targets)


def test_shapes(datasets):

    train_set, test_set = datasets

    assert train_set.data[0].shape == (1, 28, 28)
    assert test_set.data[0].shape == (1, 28, 28)


def test_values(datasets):

    train_set, test_set = datasets

    assert ((train_set.data <= 1) & (train_set.data >= 0)).all()
    assert ((test_set.data <= 1) & (test_set.data >= 0)).all()


@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_data_loader(datasets, batch_size):

    train_set, test_set = datasets

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True
    )

    train_images, train_targets = next(iter(train_loader))
    test_images, test_targets = next(iter(test_loader))

    assert train_images.shape == (batch_size, 1, 28, 28)
    assert test_images.shape == (batch_size, 1, 28, 28)

    assert train_targets.shape == (batch_size,)
    assert test_targets.shape == (batch_size,)
