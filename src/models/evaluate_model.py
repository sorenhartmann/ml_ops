import torch
from torch import nn

from src.data.make_dataset import mnist
from src.models.train_model import model_dir, test_loop


def main():

    print("Evaluating until hitting the ceiling")

    model = torch.load(model_dir / "trained_model.pt")

    _, test_set = mnist()

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True)
    test_loss, accuracy = test_loop(
        model, testloader, criterion=nn.CrossEntropyLoss(reduction="sum"))

    print(f"Test Loss: {test_loss:.3e}", end="\t")
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":

    main()
