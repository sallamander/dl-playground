"""Inception (V3) implementation written with PyTorch

Reference paper: 'Rethinking the Inception Architecture for Computer Vision'
    https://arxiv.org/abs/1512.00567

Reference Implementations:
    - https://github.com/tensorflow/models/blob/master/research/slim/nets
        /inception_v3.py
    - https://github.com/keras-team/keras-applications/blob/master
        /keras_applications/inception_v3.py
"""

from torch.nn import BatchNorm2d, Conv2d, MaxPool2d, Module, ReLU

from utils.generic_utils import validate_config


class InceptionNet(Module):
    """Inception network (V3)

    As implemented in 'Rethinking the Inception Architecture for Computer 
    Vision', referred to as Inception V3 (https://arxiv.org/abs/1512.00567).
    """

    required_config_keys = {'n_channels', 'n_classes'}

    def __init__(self, config):
        """Init

        The `config` must contain the following keys:
        - int n_channels: number of channels in the input
        - int n_classes: number of classes in the output layer

        :param config: specifies the configuration for the network
        :type config: dict
        """

        super().__init__()

        validate_config(config, self.required_config_keys)
        self.config = config

        self._set_layers()

    def _set_layers(self):
        """Set the network's layers used in the forward pass"""

        n_in_channels=self.config['n_channels']
        n_classes = self.config['n_classes']

        self.conv1 = Conv2d(
            in_channels=n_in_channels, out_channels=32,
            kernel_size=(3, 3), stride=(2, 2), bias=False
        )
        self.bn1 = BatchNorm2d(num_features=32)

        self.conv2 = Conv2d(
            in_channels=32, out_channels=32,
            kernel_size=(3, 3), stride=(1, 1), bias=False
        )
        self.bn2 = BatchNorm2d(num_features=32)
        
        self.conv3 = Conv2d(
            in_channels=32, out_channels=64,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn3 = BatchNorm2d(num_features=64)

        self.max_pooling1 = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv4 = Conv2d(
            in_channels=64, out_channels=80,
            kernel_size=(3, 3), stride=(1, 1), bias=False
        )
        self.bn4 = BatchNorm2d(num_features=64)

        self.conv5 = Conv2d(
            in_channels=80, out_channels=192,
            kernel_size=(3, 3), stride=(2, 2), bias=False
        )
        self.bn5 = BatchNorm2d(num_features=192)

        self.relu = ReLU()
