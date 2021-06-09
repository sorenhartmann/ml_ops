
import pytest
import torch
from torch.functional import broadcast_shapes

from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel


@pytest.fixture()
def batch():
    train_set, _ = mnist()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    images, targets  = next(iter(train_loader))

    return images, targets


@pytest.fixture()
def model():
    model = MyAwesomeModel()
    return model


def test_forward_shapes(batch, model):

    images, targets = batch
    out = model(images)

    # Output dimensionality 
    assert targets.shape == out.shape[:-1] 
    assert out.shape[-1] == 10
    
    # Input / output shape
    assert images.shape[:-3] == out.shape[:-1]

def test_forward_values(batch, model):

    images, targets = batch
    out = model(images)

    # Output should be different for different input
    assert not (out[0, ..., :] == out[1:, ..., :]).all()

    


