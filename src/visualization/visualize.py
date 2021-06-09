import click
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from src.data.make_dataset import mnist
from src.models.train_model import figure_dir


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
        sns.scatterplot(x=y[:, 0], y=y[:, 1], hue=classes)
        return fig


@click.command()
@click.argument("model_file", type=click.Path(exists=True))
def main(model_file):

    _, test_data = mnist()
    model = torch.load(model_file)
    model.eval()

    latent_samples = []
    classes = []
    with torch.autograd.no_grad():
        for images, labels in DataLoader(test_data, batch_size=64):
            x = model.latent_repr(images)
            latent_samples.extend(x)
            classes.extend(labels.tolist())

        latent_samples = torch.stack(latent_samples)
        y = TSNE(verbose=1, n_jobs=-1).fit_transform(latent_samples)

        fig = plt.figure()
        sns.scatterplot(x=y[:, 0], y=y[:, 1], hue=classes)
        fig.savefig(figure_dir / "tsne_mnist.pdf")


if __name__ == "__main__":

    main()
