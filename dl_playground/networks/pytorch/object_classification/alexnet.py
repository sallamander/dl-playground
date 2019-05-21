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

    def __init__(self, config):
        """Init

        `config` must contain the following keys:
        - int n_channels: number of channels of the input
        - int n_classes: number of classes in the output layer

        :param config: specifies the configuration for the network
        :type config: dict
        """

        super().__init__()

        validate_config(config, self.required_config_keys)
        self.config = config
        self._set_layers()

    def _set_layers(self):
        """Set the network's layers used in the forward pass

        This sets 5 convolutional layers (self.conv[1-5]) and 3 linear layers
        (self.linear[1-3]) in-place.
        """

        n_channels = self.config['n_channels']
        n_classes = self.config['n_classes']

        self.conv1 = Conv2d(
            in_channels=n_channels, out_channels=96,
            kernel_size=(11, 11), stride=(4, 4)
        )
        self.conv2 = Conv2d(
            in_channels=96, out_channels=256,
            kernel_size=(5, 5), padding=(2, 2)
        )
        self.conv3 = Conv2d(
            in_channels=256, out_channels=384,
            kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv4 = Conv2d(
            in_channels=384, out_channels=384,
            kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv5 = Conv2d(
            in_channels=384, out_channels=256,
            kernel_size=(3, 3), padding=(1, 1)
        )

        self.linear1 = Linear(256 * 6 * 6, out_features=4096)
        self.linear2 = Linear(in_features=4096, out_features=4096)
        self.linear3 = Linear(in_features=4096, out_features=n_classes)

    def forward(self, inputs):
        """Return the output of a forward pass of AlexNet

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_channels)
        :type inputs: torch.Tensor
        :return: outputs of an AlexNet model, of shape
         (batch_size, n_classes)
        :rtype: torch.Tensor
        """

        # === convolutional block 1 === #
        layer = self.conv1(inputs)
        layer = ReLU()(layer)
        layer = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))(layer)

        # === convolutional block 2 === #
        layer = self.conv2(layer)
        layer = ReLU()(layer)
        layer = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))(layer)

        # === convolutional blocks 3, 4, 5 === #
        layer = self.conv3(layer)
        layer = ReLU()(layer)
        layer = self.conv4(layer)
        layer = ReLU()(layer)
        layer = self.conv5(layer)
        layer = ReLU()(layer)
        layer = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))(layer)

        # === dense layers (2) === #
        layer = layer.view(layer.size(0), -1)
        layer = self.linear1(layer)
        layer = ReLU()(layer)
        layer = Dropout(0.5)(layer)
        layer = self.linear2(layer)
        layer = ReLU()(layer)
        layer = Dropout(0.5)(layer)

        # === output layer === #
        outputs = self.linear3(layer)

        return outputs
