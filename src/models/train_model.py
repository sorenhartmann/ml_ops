from pathlib import Path
import wandb
from datetime import datetime
import click
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import summary
from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from sklearn.decomposition import PCA

import seaborn as sns
sns.set_style()

model_dir = Path(__file__).parents[2] / "models"
figure_dir = Path(__file__).parents[2] / "reports" / "figures"
log_dir = Path(__file__).parents[2] / "logs"


class SummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


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


def log_test_loss(model, dataloader, tb_writer):

    criterion = nn.CrossEntropyLoss(reduction="none")

    model.eval()
    with torch.no_grad():
        losses = torch.cat(
            [criterion(model.forward(images), target) for images, target in dataloader]
        )
        tb_writer.add_histogram("Log-loss hist", losses.log())


def pca_plot(model, dataloader):

    latent_samples = [] 
    classes = []

    model.eval()
    with torch.autograd.no_grad():
        for images, labels in dataloader:

            x = model.latent_repr(images)
            latent_samples.extend(x)
            classes.extend(labels.tolist())

        latent_samples = torch.stack(latent_samples)
        y = PCA().fit_transform(latent_samples)

        fig = plt.figure()
        sns.scatterplot(x = y[:, 0], y=y[:, 1], hue=classes)
        return fig
        


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

    now = datetime.now().strftime("%b%d_%H-%M-%S")

    tb_writer = SummaryWriter(flush_secs=5, log_dir=log_dir / now)
    tb_writer.add_graph(model, next(iter(trainloader))[0])

    wandb.init(project="mnist", entity="sorenhartmann")

    config = wandb.config
    config.update(
        {
            "lr": lr,
            "dropout_perc": model.dropout_perc,
            "kernel_size_1": model.kernel_sizes[0],
            "kernel_size_2": model.kernel_sizes[1],
            "ffnn_size_1": model.layer_sizes[0],
            "ffnn_size_2": model.layer_sizes[1],
        }
    )

    wandb.watch(model)

    for e in range(epochs):

        train_loss = train_loop(model, trainloader, criterion, optimizer)

        train_losses.append(train_loss)
        tb_writer.add_scalar("Loss/Train", train_loss, e)
        wandb.log({"Loss/Train": train_loss}, step=e)

        print(f"[Epoch: {e+1:03d}/{epochs:03d}]", end="\t")
        print(f"Training Loss: {train_loss:.3e}", end="\t")

        if validate:
            test_loss, accuracy = test_loop(model, testloader, criterion)

            test_losses.append(test_loss)

            tb_writer.add_scalar("Loss/Test", test_loss, e)
            tb_writer.add_scalar("Accuracy", accuracy, e)
            wandb.log(
                {
                    "Loss/Test": test_loss,
                    "Accuracy": accuracy,
                },
                step=e,
            )

            print(f"Test Loss: {test_loss:.3e}", end="\t")
            print(f"Accuracy: {accuracy:.3f}", end="\t")

        print(flush=True)

    plt.plot(range(1, epochs + 1), train_losses, label="Train loss")
    if validate:

        plt.plot(range(1, epochs + 1), test_losses, label="Test loss")

        tb_writer.add_hparams(
            {
                "lr": lr,
                "dropout_perc": model.dropout_perc,
                "kernel_size_1": model.kernel_sizes[0],
                "kernel_size_2": model.kernel_sizes[1],
                "ffnn_size_1": model.layer_sizes[0],
                "ffnn_size_2": model.layer_sizes[1],
            },
            {"accuracy": accuracy},
        )

        log_test_loss(model, testloader, tb_writer)
        fig = pca_plot(model, testloader)
        wandb.log({"Latent PCA": wandb.Image(fig)})


    tb_writer.close()
    wandb.finish()

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    figure_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(figure_dir / "loss_curves.pdf")

    model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model, model_dir / "trained_model.pt")


if __name__ == "__main__":

    main()
