from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel
from src.models.train_model import train_loop, train_step, test_loop as t_loop
from unittest.mock import MagicMock

import torch
from torch import nn, optim
import pytest


@pytest.fixture()
def batch():
    train_set, _ = mnist()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    images, targets = next(iter(train_loader))

    return images, targets


@pytest.fixture()
def data_loader():
    train_set, _ = mnist()

    dataset = torch.utils.data.Subset(train_set, range(100))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    return data_loader


@pytest.fixture()
def model():
    model = MyAwesomeModel()
    return model


def test_train_step(batch, model):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    optimizer.zero_grad = MagicMock(side_effect=optimizer.zero_grad)
    model.train = MagicMock(side_effect=model.train)

    images, targets = batch

    old_parameters = [p.clone() for p in model.parameters()]
    train_step(model, images, targets, criterion, optimizer)

    # All parameters should be different after step
    assert all(~torch.equal(a, b)
               for a, b in zip(old_parameters, model.parameters()))

    # Zero grad should be called
    optimizer.zero_grad.assert_called()


def test_test_loop(data_loader, model):

    criterion = nn.CrossEntropyLoss(reduction="sum")

    model.eval = MagicMock(side_effect=model.eval)

    old_parameters = [p.clone() for p in model.parameters()]
    t_loop(model, data_loader, criterion)

    # All parameters should be the same after loop
    assert all(torch.equal(a, b)
               for a, b in zip(old_parameters, model.parameters()))

    # Evaluate mode should be called
    model.eval.assert_called()


def test_train_loop(data_loader, model):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    model.train = MagicMock(side_effect=model.train)

    old_parameters = [p.clone() for p in model.parameters()]
    train_loop(model, data_loader, criterion, optimizer)

    # All parameters should be different after step
    assert all(~torch.equal(a, b)
               for a, b in zip(old_parameters, model.parameters()))

    # Evaluate mode should be called
    model.train.assert_called()
