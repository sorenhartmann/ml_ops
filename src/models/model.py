from torch import nn


def compute_conv_dim(dim_size, kernel_size, stride=1, padding=0):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)

class MyAwesomeModel(nn.Module):

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

    dropout_perc = 0.5

    def __init__(self):

        super().__init__()

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
            conv_layers.append(nn.Dropout2d(p=self.dropout_perc))
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
            ffnn_layers.append(nn.Dropout(self.dropout_perc))
            ffnn_layers.append(nn.ReLU())
            in_features = out_features
        ffnn_layers.append(nn.Linear(in_features, self.output_size))

        self.ffnn = nn.Sequential(*ffnn_layers)

    def forward(self, x):
        # make sure input tensor is flattened

        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.ffnn(x)

        return x

    def latent_repr(self, x):

        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.ffnn[:-3](x)

        return x