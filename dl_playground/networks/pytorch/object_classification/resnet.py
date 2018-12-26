"""ResNet implementation"""

# TODO:
# - Add reference
# - Check to see if the paper implementation includes / excludes the use of a
# bias term in the bottleneck convs
# - Finish the IdentityShorcut block
# - Add in code for the projection shorcut

from torch.nn import BatchNorm2d, Conv2d, Module, ReLU

from utils.generic_utils import validate_config


class BottleneckBlock(Module):
    """Bottleneck Block"""

    def __init__(self, n_in_channels, n_out_channels):

        self.conv1 = Conv2d(
            in_channels=n_in_channels, out_channels=n_out_channels,
            kernel_size=(1, 1), stride=(1, 1)
        )
        self.bn1 = BatchNorm2d(num_features=n_out_channels)

        self.conv2 = Conv2d(
            in_channels=n_out_channels, out_channels=n_out_channels,
            kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)
        )
        self.bn2 = BatchNorm2d(num_features=n_out_channels)

        self.conv3 = Conv2d(
            in_channels=n_out_channels, out_channels=(n_out_channels * 4),
            kernel_size=(1, 1)
        )
        self.bn3 = BatchNorm2d(num_features=(n_out_channels * 4))

        self.relu = ReLU()

    def foward(self, inputs):

        layer = self.conv1(inputs)
        layer = self.bn1(layer)
        layer = self.relu(layer)

        layer = self.conv2(layer)
        layer = self.bn2(layer)
        layer = self.relu(layer)

        layer = self.conv3(layer)
        layer = self.bn3(layer)

        layer += inputs
        layer = self.relu(layer)

        return layer


# TODO: This is not finished
class IdentityShorcut(Module)

    def __init__(self, n_in_channels, n_out_channels):

        self.conv1 = Conv2d(
            in_channels=n_in_channels, out_channels=n_out_channels,
            kernel_size=(3, 3), strides=(2, 2), padding=(1, 1)
        )
        self.bn1 = BatchNorm2d(num_features=n_out_channels)

        self.conv2 = Conv2d(
            in_channels=n_out_channels, out_channels=n_out_channels,
            kernel_size=(3, 3), padding=(1, 1)
        )
        self.bn2 = BatchNorm2d(num_features=n_out_channels)


class ResidualBlock(Module):


    def __init__(self, n_in_channels, n_out_channels):

        self.conv1 = Conv2d(
            in_channels=n_in_channels, out_channels=n_out_channels,
            kernel_size=(3, 3), padding=(1, 1)
        )
        self.bn1 = BatchNorm2d(num_features=n_out_channels)
        
        self.conv2 = Conv2d(
            in_channels=n_out_channels, out_channels=n_out_channels,
            kernel_size=(3, 3), padding=(1, 1)
        )
        self.bn2 = BatchNorm2d(num_features=n_out_channels)

        self.relu = ReLU()

    def forward(self, inputs):

        layer = self.conv1(layer)
        layer = self.bn1(layer)
        layer = self.relu(layer)

        layer = self.conv2(layer)
        layer = self.bn2(layer)
        layer = self.relu(layer)

        layer += inputs
        layer = self.relu(layer)
        
        return layer


class ResNet(object):
    """ResNet network"""

    required_config_keys = {'n_channels', 'n_classes'}

    def __init__(self, config):
        """Init

        The `config` must contain the following keys:
        - int n_channels: number of channels of the input
        - int n_classes: number of classes in the output layer

        :param config: specifies the configuration for the network
        :type config: dict
        """

        validate_config(config, self.required_config_keys)
        self.config = config
        self._set_layers()

    def forward(self, inputs):
        """Return the output of a forward pass of the specified ResNet

        :param inputs: TODO
	:type inputs: TODO
        :return: TODO
	:rtype:
        """
        

        
