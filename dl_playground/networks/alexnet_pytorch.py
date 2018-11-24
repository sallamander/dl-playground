"""AlexNet implementation written with torch.nn

Reference paper (using bitly link to save line length):
https://bit.ly/2v4Aihl

The main difference between this implementation and the paper is that it does
not use any parallelization across GPUs, nor include a local response layer.
"""

from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU

from utils.generic_utils import validate_config


class AlexNet(Module):
    """AlexNet model"""

    required_config_keys = {'n_channels', 'n_classes'}

    def __init__(self, network_config):
        """Init

        `network_config` must contain the following keys:
        - int n_channels: number of channels of the input
        - int n_classes: number of classes in the output layer

        :param network_config: specifies the configuration for the network
        :type network_config: dict
        """

        super().__init__()

        validate_config(network_config, self.required_config_keys)
        self.network_config = network_config

    def forward(self, inputs):
        """Return the output of a forward pass of AlexNet

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_channels)
        :type inputs: torch.Tensor
        :return: outputs of an AlexNet model, of shape
         (batch_size, n_classes)
        :rtype: torch.Tensor
        """

        n_channels = self.network_config['n_channels']
        n_classes = self.network_config['n_classes']

        # (batch_size, height, width, n_channels) =>
        # (batch_size, n_channels, height, width)
        inputs = inputs.transpose(1, 3)

        # === convolutional block 1 === #
        layer = Conv2d(
            in_channels=n_channels, out_channels=96,
            kernel_size=(11, 11), stride=(4, 4)
        )(inputs)
        layer = ReLU()(layer)
        layer = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))(layer)

        # === convolutional block 2 === #
        layer = Conv2d(
            in_channels=96, out_channels=256, kernel_size=(5, 5)
        )(layer)
        layer = ReLU()(layer)
        layer = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))(layer)

        # === convolutional blocks 3, 4, 5 === #
        layer = Conv2d(
            in_channels=256, out_channels=384, kernel_size=(3, 3)
        )(layer)
        layer = ReLU()(layer)
        layer = Conv2d(
            in_channels=384, out_channels=384, kernel_size=(3, 3)
        )(layer)
        layer = ReLU()(layer)
        layer = Conv2d(
            in_channels=384, out_channels=256, kernel_size=(3, 3)
        )(layer)
        layer = ReLU()(layer)
        layer = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))(layer)

        # === dense layers (2) === #
        layer = layer.view(layer.size(0), -1)
        layer = Linear(in_features=layer.size(1), out_features=4096)(layer)
        layer = ReLU()(layer)
        layer = Dropout(0.5)(layer)
        layer = Linear(in_features=4096, out_features=4096)(layer)
        layer = ReLU()(layer)
        layer = Dropout(0.5)(layer)

        # === output layer === #
        outputs = Linear(in_features=4096, out_features=n_classes)(layer)

        return outputs
