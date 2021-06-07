import sys
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, dataloader
from torch import nn, optim
from tqdm import tqdm

import sys
from pathlib import Path

root_dir = Path(__file__).parent.resolve()

sys.path.append(root_dir)

from data import mnist  # type: ignore
from model import MyAwesomeModel  # type: ignore


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


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):

        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--epochs", default=50, type=int)
        parser.add_argument("--validate", default=True, type=bool)
        parser.add_argument("--out-file", default="model.pt", type=str)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])

        model = MyAwesomeModel()
        train_set, test_set = mnist()

        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=64, shuffle=True
        )
        if args.validate:
            testloader = torch.utils.data.DataLoader(
                test_set, batch_size=64, shuffle=True
            )

        criterion = nn.CrossEntropyLoss(reduction="sum")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_losses = []
        test_losses = []

        for e in range(args.epochs):

            train_loss = train_loop(model, trainloader, criterion, optimizer)
            train_losses.append(train_loss)

            print(f"[Epoch: {e+1:03d}/{args.epochs:03d}]", end="\t")
            print(f"Training Loss: {train_loss:.3e}", end="\t")

            if args.validate:
                test_loss, accuracy = test_loop(model, testloader, criterion)
                test_losses.append(test_loss)

                print(f"Test Loss: {test_loss:.3e}", end="\t")
                print(f"Accuracy: {accuracy:.3f}", end="\t")

            print(flush=True)

        plt.plot(range(1, args.epochs + 1), train_losses, label="Train loss")
        if args.validate:
            plt.plot(range(1, args.epochs + 1), test_losses, label="Test loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        torch.save(model, root_dir / args.out_file)

    def evaluate(self):

        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--file", default="model.pt")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])

        if args.file:
            model = torch.load(root_dir / args.file)

        _, test_set = mnist()

        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        test_loss, accuracy = test_loop(model, testloader, criterion=nn.CrossEntropyLoss(reduction="sum"))

        print(f"Test Loss: {test_loss:.3e}", end="\t")
        print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":

    TrainOREvaluate()
