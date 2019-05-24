"""ResNet implementation written with PyTorch

Reference paper: https://arxiv.org/abs/1512.03385

Main Reference Implementation:
    - https://github.com/KaimingHe/deep-residual-networks/blob
        /master/prototxt/ResNet-50-deploy.prototxt

Other Reference Implementations:

    - https://github.com/taehoonlee/tensornets/blob/master/tensornets
        /resnets.py
    - https://github.com/keras-team/keras-applications/blob/master
        /keras_applications/resnet50.py

This implementation follows the "Main Reference Implementation" as closely as
possible, since it is the paper author's implementation. The "Other Reference
Implementations" were used as a reference, and appear to have slight
differences from the main implementation (although the `tensornets`
implementation is extremely close).
"""

import torch
from torch.nn import (
    AvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ModuleList, ReLU
)

from utils.generic_utils import validate_config


class BottleneckBlock(Module):
    """Bottleneck Residual Block"""

    def __init__(self, n_in_channels, n_out_channels):
        """Init

        :param n_in_channels: number of channels in the inputs passed to the
         `forward` method
        :type n_in_channels: int
        :param n_out_channels: number of output channels (convolutional
         filters) to use in the convolutions applied to the inputs passed to
         the `forward` method
        :type n_out_channels: int
        """

        super().__init__()

        self.conv1 = Conv2d(
            in_channels=n_in_channels, out_channels=n_out_channels,
            kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.bn1 = BatchNorm2d(num_features=n_out_channels)

        self.conv2 = Conv2d(
            in_channels=n_out_channels, out_channels=n_out_channels,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn2 = BatchNorm2d(num_features=n_out_channels)

        self.conv3 = Conv2d(
            in_channels=n_out_channels, out_channels=(n_out_channels * 4),
            kernel_size=(1, 1), bias=False
        )
        self.bn3 = BatchNorm2d(num_features=(n_out_channels * 4))

        self.relu = ReLU()

    def forward(self, inputs):
        """Return the output of a forward pass of a BottleneckBlock

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_in_channels)
        :type inputs: torch.Tensor
        :return: outputs of a BottleneckBlock, of shape
         (batch_size, height, width, n_out_channels * 4)
        :rtype: torch.Tensor
        """

        layer = self.conv1(inputs)
        layer = self.bn1(layer)
        layer = self.relu(layer)

        layer = self.conv2(layer)
        layer = self.bn2(layer)
        layer = self.relu(layer)

        layer = self.conv3(layer)
        layer = self.bn3(layer)

        layer = torch.add(layer, inputs)
        layer = self.relu(layer)

        return layer


class ProjectionShortcut(Module):
    """ProjectionShortcut module

    This module is inserted at the beginning of each residual stage, and
    downsamples the input 2x using a convolution with stride 2 (except for the
    first residual stage, in which there is no downsampling).
    """

    def __init__(self, n_in_channels, n_out_channels, stride):
        """Init

        :param n_in_channels: number of channels in the inputs passed to the
         `forward` method
        :type n_in_channels: int
        :param n_out_channels: number of output channels (convolutional
         filters) to use in the convolutions applied to the inputs passed to
         the `forward` method
        :type n_out_channels: int
        :param stride: holds the stride to use in the first convolution of the
         block (i.e. for downsampling)
        :type stride: tuple(int)
        """

        super().__init__()

        self.conv1 = Conv2d(
            in_channels=n_in_channels, out_channels=n_out_channels,
            kernel_size=(1, 1), stride=stride, bias=False
        )
        self.bn1 = BatchNorm2d(num_features=n_out_channels)

        self.conv2 = Conv2d(
            in_channels=n_out_channels, out_channels=n_out_channels,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn2 = BatchNorm2d(num_features=n_out_channels)

        self.conv3 = Conv2d(
            in_channels=n_out_channels, out_channels=(n_out_channels * 4),
            kernel_size=(1, 1), bias=False
        )
        self.bn3 = BatchNorm2d(num_features=(n_out_channels * 4))

        self.projection_conv = Conv2d(
            in_channels=n_in_channels, out_channels=(n_out_channels * 4),
            kernel_size=(1, 1), stride=stride, bias=False
        )
        self.projection_bn = BatchNorm2d(num_features=(n_out_channels * 4))

        self.relu = ReLU()

    def forward(self, inputs):
        """Return the output of a forward pass of a ProjectionShortcut

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_in_channels)
        :type inputs: torch.Tensor
        :return: outputs of a BottleneckBlock, of shape
         (batch_size, height / 2, width / 2, n_out_channels * 4)
        :rtype: torch.Tensor
        """

        layer = self.conv1(inputs)
        layer = self.bn1(layer)
        layer = self.relu(layer)

        layer = self.conv2(layer)
        layer = self.bn2(layer)
        layer = self.relu(layer)

        layer = self.conv3(layer)
        layer = self.bn3(layer)

        shortcut = self.projection_conv(inputs)
        shortcut = self.projection_bn(shortcut)

        layer = torch.add(layer, shortcut)
        layer = self.relu(layer)

        return layer


class ResNet(Module):
    """ResNet network"""

    required_config_keys = {
        'n_channels', 'n_classes', 'n_initial_channels', 'n_blocks_per_stage'
    }

    def __init__(self, config):
        """Init

        The `config` must contain the following keys:
        - int n_channels: number of channels of the input
        - int n_classes: number of classes in the output layer
        - int n_initial_channels: number of filters in the initial convolution
          as well as the first residual block; subsequent residual blocks will
          have 2x the filters as the previous residual block
        - list[int] n_blocks_per_stage: iterable holding the number of residual
          blocks to use per residual stage; the length of n_blocks_per_stage
          defines the number of residual stages in the network

        :param config: specifies the configuration for the network
        :type config: dict
        """

        super().__init__()

        validate_config(config, self.required_config_keys)
        self.config = config

        self._set_layers()

    def _set_layers(self):
        """Set the network's layers used in the forward pass

        This sets the following layers in-place:
        - A general relu used throughout the forward pass
        - Initial conv, bn, and max pooling layers
        - Residual stages, each consisting of a ProjectionShortcut module and
          some number of BottleneckBlock modules. The number of residual stages
          depends on the length of the `n_blocks_per_stage` iterable in
          self.config.
        - Global average pooling layer
        - Final linear layer
        """

        n_initial_channels = self.config['n_initial_channels']
        n_blocks_per_stage = self.config['n_blocks_per_stage']

        self.conv = Conv2d(
            in_channels=3, out_channels=n_initial_channels, kernel_size=(7, 7),
            stride=(2, 2), padding=(3, 3)
        )
        self.bn = BatchNorm2d(num_features=n_initial_channels)
        self.relu = ReLU()
        self.max_pooling = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.residual_stages = ModuleList()
        n_out_channels = n_initial_channels
        for idx_stage, n_blocks in enumerate(n_blocks_per_stage):
            stage_blocks = ModuleList()

            # Each residual block ends with a conv using 4x the number of
            # n_out_channels, which requires us to adjust n_in_channels to
            # account for that. The exception is when we first enter the first
            # residual stage, where the number of channels is simply
            # n_out_channels.
            if idx_stage == 0:
                n_in_channels = n_out_channels
                stride = (1, 1)
            else:
                # Multiply by 2 instead of 4 because at the end of the current
                # iteration of the for loop the `n_out_channels` will be
                # multiplied by 2, which accounts for half of the 4x increase
                # in n_in_channels
                n_in_channels = n_out_channels * 2
                stride = (2, 2)

            stage_blocks.append(ProjectionShortcut(
                n_in_channels, n_out_channels, stride
            ))

            for _ in range(n_blocks - 1):
                # n_in_channels passed to BottleneckBlock needs to be 4x
                # n_out_channels because each ProjectionShortcut ends with a
                # conv using 4x the number of n_out_channels
                stage_blocks.append(BottleneckBlock(
                    n_out_channels * 4, n_out_channels
                ))
            self.residual_stages.append(stage_blocks)

            n_out_channels *= 2

        self.average_pooling = AvgPool2d(kernel_size=(7, 7))

        # Divide n_out_channels by 2 to negate the last multiplication in the
        # for loop that builds the residual stages
        n_out_channels = n_out_channels // 2 * 4
        self.linear = Linear(
            in_features=n_out_channels, out_features=1000
        )

    def forward(self, inputs):
        """Return the output of a forward pass of the specified ResNet

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_in_channels)
        :type inputs: torch.Tensor
        :return: outputs of a forward pass of ResNet, of shape
         (batch_size, n_classes)
        :rtype: torch.Tensor
        """

        layer = self.conv(inputs)
        layer = self.bn(layer)
        layer = self.relu(layer)
        layer = self.max_pooling(layer)

        for residual_stage in self.residual_stages:
            for residual_block in residual_stage:
                layer = residual_block(layer)

        layer = self.average_pooling(layer)
        layer = layer.reshape(layer.size(0), -1)
        layer = self.linear(layer)

        return layer
