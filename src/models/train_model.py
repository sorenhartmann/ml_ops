from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel
from torch import nn, optim

model_dir = Path(__file__).parents[2] / "models"
figure_dir = Path(__file__).parents[2] / "reports" / "figures"


def train_loop(model, dataloader, criterion, optimizer):

    train_loss = 0

    model.train()
    for images, labels in dataloader:

        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(dataloader.dataset)

    return train_loss


def test_loop(model, dataloader, criterion):

    correct = 0
    total = 0
    test_loss = 0

    model.eval()
    with torch.no_grad():

        for images, labels in dataloader:

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = labels.data == ps.max(1)[1]

            correct += equality.sum()
            total += len(labels)

    accuracy = correct / total
    test_loss /= len(dataloader.dataset)

    return test_loss, accuracy


@click.command()
@click.option("--lr", default=0.001, type=float)
@click.option("--epochs", default=10, type=int)
@click.option("--validate", default=True, type=bool)
def main(lr, epochs, validate):

    print("Training day and night")

    model = MyAwesomeModel()
    train_set, test_set = mnist()

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    if validate:
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for e in range(epochs):

        train_loss = train_loop(model, trainloader, criterion, optimizer)
        train_losses.append(train_loss)

        print(f"[Epoch: {e+1:03d}/{epochs:03d}]", end="\t")
        print(f"Training Loss: {train_loss:.3e}", end="\t")

        if validate:
            test_loss, accuracy = test_loop(model, testloader, criterion)
            test_losses.append(test_loss)

            print(f"Test Loss: {test_loss:.3e}", end="\t")
            print(f"Accuracy: {accuracy:.3f}", end="\t")

        print(flush=True)

    plt.plot(range(1, epochs + 1), train_losses, label="Train loss")
    if validate:
        plt.plot(range(1, epochs + 1), test_losses, label="Test loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    figure_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figure_dir / "loss_curves.pdf")

    model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model, model_dir / "trained_model.pt")


if __name__ == "__main__":

    main()
