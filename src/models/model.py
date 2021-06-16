from torch import nn
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F


def compute_conv_dim(dim_size, kernel_size, stride=1, padding=0):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)


class MyAwesomeModel(pl.LightningModule):

    input_channels = 1
    input_size = (28, 28)

    # Convolutional model
    kernel_sizes = [5, 3]
    out_channels = [5, 5]
    stride = [1, 1]
    padding = [2, 1]

    # FFNN
    layer_sizes = [256, 128]
    output_size = 10

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MyAwesomeModel")
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--dropout_perc', type=float, default=0.50)
        return parent_parser


    def __init__(self, lr, dropout_perc, **kwargs):

        super().__init__()

        self.save_hyperparameters("lr", "dropout_perc")

        self.lr = lr

        conv_layers = []
        in_channels = self.input_channels
        height, width = self.input_size

        for kernel_size, out_channels, stride, padding in zip(
            self.kernel_sizes, self.out_channels, self.stride, self.padding
        ):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.Dropout2d(p=dropout_perc))
            conv_layers.append(nn.ReLU())

            height = compute_conv_dim(height, kernel_size, stride, padding)
            width = compute_conv_dim(width, kernel_size, stride, padding)

            in_channels = out_channels

        max_pool_kernel_size = 3
        conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_kernel_size))
        height = compute_conv_dim(
            height, max_pool_kernel_size, stride=max_pool_kernel_size
        )
        width = compute_conv_dim(
            width, max_pool_kernel_size, stride=max_pool_kernel_size
        )

        self.cnn = nn.Sequential(*conv_layers)

        ffnn_layers = []
        in_features = height * width * out_channels

        for out_features in self.layer_sizes:
            ffnn_layers.append(nn.Linear(in_features, out_features))
            ffnn_layers.append(nn.Dropout(dropout_perc))
            ffnn_layers.append(nn.ReLU())
            in_features = out_features
        ffnn_layers.append(nn.Linear(in_features, self.output_size))

        self.ffnn = nn.Sequential(*ffnn_layers)

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # make sure input tensor is flattened

        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.ffnn(x)

        return x

    def training_step(self, batch, batch_index):

        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_index):

        images, labels = batch
        outputs = self(images)

        loss = F.cross_entropy(outputs, labels)
        predictions = torch.max(outputs, 1)[1]

        self.log("val_loss", loss)
        self.log("val_accuracy", self.accuracy(predictions, labels), prog_bar=True)

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def latent_repr(self, x):

        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.ffnn[:-3](x)

        return x
