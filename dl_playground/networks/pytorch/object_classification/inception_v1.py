"""Inception (V1) implementation written with PyTorch

Reference paper: 'Going Deeper with Convolutions'
    https://arxiv.org/abs/1409.4842

Reference Implementations:
    (1) https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet
        /train_val.prototxt
    (2) https://github.com/soeaver/caffe-model/blob/master/cls/inception
        /deploy_inception-v1-dsd.prototxt
    (3) https://github.com/tensorflow/tensorflow/blob/master/tensorflow
        /contrib/slim/python/slim/nets/inception_v1.py
"""

import torch
from torch.nn import (
    AvgPool2d, Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU
)
from torch.nn.init import xavier_uniform_, constant_

from trainet.utils.generic_utils import validate_config


class AuxiliaryClassifier(Module):
    """Auxiliary Classifier

    As implemented in 'Going Deeper with Convolutions', the original Inception
    paper (https://arxiv.org/abs/1409.4842).
    """

    def __init__(self, n_in_channels, n_classes):
        """Init

        :param n_in_channels: number of channels in the inputs passed to the
         `forward` method
        :type n_in_channels: int
        :param n_classes: number of classes to predict from the final linear
         layer
        :type n_classes: int
        """

        super().__init__()

        self.average_pooling = AvgPool2d(kernel_size=(5, 5), stride=(3, 3))
        self.conv = Conv2d(
            in_channels=n_in_channels, out_channels=128,
            kernel_size=(1, 1), stride=(1, 1)
        )
        self.linear1 = Linear(in_features=(128 * 4 * 4), out_features=1024)
        self.dropout = Dropout(0.70)
        self.linear2 = Linear(in_features=1024, out_features=n_classes)

        self.relu = ReLU()

    def forward(self, inputs):
        """Return the output of the AuxiliaryClassifier

        :param inputs: batch of inputs, of shape
         (batch_size, n_in_channels, height, width)
        :type inputs: torch.Tensor
        :return: outputs of an AuxiliaryClassifier, of shape
         (batch_size, n_classes)
        :rtype: torch.Tensor
        """

        layer = self.average_pooling(inputs)
        layer = self.conv(layer)
        layer = self.relu(layer)

        layer = layer.reshape(layer.size(0), -1)
        layer = self.linear1(layer)
        layer = self.relu(layer)

        layer = self.dropout(layer)
        outputs = self.linear2(layer)

        return outputs


class InceptionModule(Module):
    """Inception Module

    As implemented in 'Going Deeper with Convolutions', the original Inception
    paper (https://arxiv.org/abs/1409.4842).
    """

    def __init__(self, n_in_channels, n_11_channels, n_33_reduce_channels,
                 n_33_channels, n_55_reduce_channels, n_55_channels,
                 n_post_pool_channels):
        """Init

        :param n_in_channels: number of channels in the inputs passed to the
         `forward` method
        :type n_in_channels: int
        :param n_11_channels: number of channels to use in the 1x1 convolution
         applied to the input
        :type n_11_channels: int
        :param n_33_reduce_channels: number of channels to use in the 1x1
         convolution applied to the input to reduce the number of channels
         before a 3x3 convolution
        :type n_33_reduce_channels: int
        :param n_33_channels: number of channels to use in the 3x3 convolution
         applied to the input
        :type n_33_channels: int
        :param n_55_reduce_channels: number of channels to use in the 1x1
         convolution applied to the input to reduce the number of channels
         before a 5x5 convolution
        :type n_55_reduce_channels: int
        :param n_55_channels: number of channels to use in the 5x5 convolution
         applied to the input
        :type n_55_channels: int
        :param n_post_pool_channels: number of channels to use in the
         convolution applied to the output of the 3x3 pooling
        :type n_post_pool_channels: int
        """

        super().__init__()

        self.conv_11 = Conv2d(
            in_channels=n_in_channels, out_channels=n_11_channels,
            kernel_size=(1, 1), stride=(1, 1)
        )
        self.conv_33_reduce = Conv2d(
            in_channels=n_in_channels, out_channels=n_33_reduce_channels,
            kernel_size=(1, 1), stride=(1, 1)
        )
        self.conv_55_reduce = Conv2d(
            in_channels=n_in_channels, out_channels=n_55_reduce_channels,
            kernel_size=(1, 1), stride=(1, 1)
        )

        self.conv_33 = Conv2d(
            in_channels=n_33_reduce_channels, out_channels=n_33_channels,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_55 = Conv2d(
            in_channels=n_55_reduce_channels, out_channels=n_55_channels,
            kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)
        )

        self.max_pooling = MaxPool2d(
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv_post_pooling = Conv2d(
            in_channels=n_in_channels, out_channels=n_post_pool_channels,
            kernel_size=(1, 1), stride=(1, 1)
        )

        self.relu = ReLU()

    def forward(self, inputs):
        """Return the output of a forward pass of an InceptionModule

        :param inputs: batch of inputs, of shape
         (batch_size, n_in_channels, height, width)
        :type inputs: torch.Tensor
        :return: outputs of a InceptionModule, of shape
         (batch_size, n_out_channels, height, width),
         where n_out_channels=
            (n_11_channels + n_33_channels +
             n_55_channels + n_post_pool_channels)
        :rtype: torch.Tensor
        """

        layer_11 = self.conv_11(inputs)
        layer_11 = self.relu(layer_11)

        layer_33_reduce = self.conv_33_reduce(inputs)
        layer_33_reduce = self.relu(layer_33_reduce)
        layer_33 = self.conv_33(layer_33_reduce)
        layer_33 = self.relu(layer_33)

        layer_55_reduce = self.conv_55_reduce(inputs)
        layer_55_reduce = self.relu(layer_55_reduce)
        layer_55 = self.conv_55(layer_55_reduce)
        layer_55 = self.relu(layer_55)

        layer_pooled = self.max_pooling(inputs)
        layer_post_pooling = self.conv_post_pooling(layer_pooled)
        layer_post_pooling = self.relu(layer_post_pooling)

        outputs = torch.cat(
            (layer_11, layer_33, layer_55, layer_post_pooling), dim=1
        )
        return outputs


class InceptionNet(Module):
    """Inception network

    As implemented in 'Going Deeper with Convolutions', the original Inception
    paper (https://arxiv.org/abs/1409.4842).
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

        self._initialize_layers()
        self._initialize_layer_weights()

        self.n_outputs = 3

    def _initialize_layers(self):
        """Set the network's layers used in the forward pass"""

        n_in_channels = self.config['n_channels']
        n_classes = self.config['n_classes']

        self.conv1 = Conv2d(
            in_channels=n_in_channels, out_channels=64,
            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
        )
        self.max_pooling1 = MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )

        self.conv2 = Conv2d(
            in_channels=64, out_channels=192,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv3 = Conv2d(
            in_channels=192, out_channels=192,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.max_pooling2 = MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )

        self.inception_3a = InceptionModule(
            n_in_channels=192, n_11_channels=64, n_33_reduce_channels=96,
            n_33_channels=128, n_55_reduce_channels=16, n_55_channels=32,
            n_post_pool_channels=32
        )
        self.inception_3b = InceptionModule(
            n_in_channels=256, n_11_channels=128, n_33_reduce_channels=128,
            n_33_channels=192, n_55_reduce_channels=32, n_55_channels=96,
            n_post_pool_channels=64
        )
        self.max_pooling3 = MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )

        self.inception_4a = InceptionModule(
            n_in_channels=480, n_11_channels=192, n_33_reduce_channels=96,
            n_33_channels=208, n_55_reduce_channels=16, n_55_channels=48,
            n_post_pool_channels=64
        )
        self.inception_4b = InceptionModule(
            n_in_channels=512, n_11_channels=160, n_33_reduce_channels=112,
            n_33_channels=224, n_55_reduce_channels=24, n_55_channels=64,
            n_post_pool_channels=64
        )
        self.inception_4c = InceptionModule(
            n_in_channels=512, n_11_channels=128, n_33_reduce_channels=128,
            n_33_channels=256, n_55_reduce_channels=24, n_55_channels=64,
            n_post_pool_channels=64
        )
        self.inception_4d = InceptionModule(
            n_in_channels=512, n_11_channels=112, n_33_reduce_channels=144,
            n_33_channels=288, n_55_reduce_channels=32, n_55_channels=64,
            n_post_pool_channels=64
        )
        self.inception_4e = InceptionModule(
            n_in_channels=528, n_11_channels=256, n_33_reduce_channels=160,
            n_33_channels=320, n_55_reduce_channels=32, n_55_channels=128,
            n_post_pool_channels=128
        )
        self.max_pooling4 = MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )

        self.inception_5a = InceptionModule(
            n_in_channels=832, n_11_channels=256, n_33_reduce_channels=160,
            n_33_channels=320, n_55_reduce_channels=32, n_55_channels=128,
            n_post_pool_channels=128
        )
        self.inception_5b = InceptionModule(
            n_in_channels=832, n_11_channels=384, n_33_reduce_channels=192,
            n_33_channels=384, n_55_reduce_channels=48, n_55_channels=128,
            n_post_pool_channels=128
        )

        self.average_pooling = AvgPool2d(kernel_size=(7, 7))
        self.dropout = Dropout(0.4)
        self.linear = Linear(
            in_features=1024, out_features=n_classes
        )

        self.auxiliary_classifier1 = AuxiliaryClassifier(
            n_in_channels=512, n_classes=n_classes
        )
        self.auxiliary_classifier2 = AuxiliaryClassifier(
            n_in_channels=528, n_classes=n_classes
        )

        self.relu = ReLU()

    def _initialize_layer_weights(self):
        """Initialize the weights of the network's layers

        This follows the initialization strategy in reference implementation
        (1), which uses Xavier / Glorot uniform initialization, with biases
        being initialized with a constant of 0.2.
        """

        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0.2)

    def forward(self, inputs):
        """Return the output of a forward pass of InceptionNet

        :param inputs: batch of input images, of shape
         (batch_size, n_in_channels, height, width)
        :type inputs: torch.Tensor
        :return: outputs of a forward pass of ResNet, of shape
         (batch_size, n_classes)
        :rtype: torch.Tensor
        """

        layer = self.conv1(inputs)
        layer = self.relu(layer)
        layer = self.max_pooling1(layer)

        layer = self.conv2(layer)
        layer = self.relu(layer)
        layer = self.conv3(layer)
        layer = self.relu(layer)
        layer = self.max_pooling2(layer)

        layer = self.inception_3a(layer)
        layer = self.inception_3b(layer)
        layer = self.max_pooling3(layer)

        layer_4a = self.inception_4a(layer)
        layer = self.inception_4b(layer_4a)
        layer = self.inception_4c(layer)
        layer_4d = self.inception_4d(layer)
        layer = self.inception_4e(layer_4d)
        layer = self.max_pooling4(layer)

        layer = self.inception_5a(layer)
        layer = self.inception_5b(layer)
        layer = self.average_pooling(layer)

        layer = layer.reshape(layer.size(0), -1)
        layer = self.dropout(layer)
        outputs = self.linear(layer)

        auxiliary_outputs1 = self.auxiliary_classifier1(layer_4a)
        auxiliary_outputs2 = self.auxiliary_classifier2(layer_4d)

        return outputs, auxiliary_outputs1, auxiliary_outputs2
