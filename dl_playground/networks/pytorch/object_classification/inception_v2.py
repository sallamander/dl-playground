"""Inception (V2) implementation written with PyTorch

Reference paper: 'Batch Normalization: Accelerating Deep 
    Network Training by Reducing Internal Covariate Shift'
    https://arxiv.org/abs/1502.03167

Reference Implementations:
    - https://github.com/lim0606/caffe-googlenet-bn/blob/master
        /train_val.prototxt
    - https://github.com/tensorflow/models/blob/master/research/slim/nets
        /inception_v2.py
"""

from ktorch.layers import SeparableConv2d
import torch
from torch.nn import (
    AvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d, MaxPool2d, Module, ReLU
)

from utils.generic_utils import validate_config


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
            kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.bn_conv = BatchNorm2d(num_features=128)
        self.linear1 = Linear(in_features=128, out_features=1024)
        self.bn_linear = BatchNorm1d(num_features=1024)
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
        layer = self.bn_conv(layer)
        layer = self.relu(layer)

        layer = layer.reshape(layer.size(0), -1)
        layer = self.linear1(layer)
        layer = self.bn_linear(layer)
        layer = self.relu(layer)

        layer = self.dropout(layer)
        outputs = self.linear(layer)

        return outputs


class InceptionModule(Module):
    """Inception Module

    As implemented in 'Going Deeper with Convolutions', the original Inception
    paper (https://arxiv.org/abs/1409.4842).
    """

    def __init__(self, n_in_channels, n_11_channels, n_33_reduce_channels,
                 n_33_channels, n_3333_reduce_channels, n_3333_channels,
                 n_post_pool_channels, pooling_type, downsample):
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
        :param n_3333_reduce_channels: number of channels to use in the 1x1
         convolution applied to the input to reduce the number of channels
         before the double 3x3 convolution
        :type n_3333_reduce_channels: int
        :param n_3333_channels: number of channels to use in the double 3x3
         convolution applied to the input
        :type n_3333_channels: int
        :param n_post_pool_channels: number of channels to use in the
         convolution applied to the output of the 3x3 pooling
        :type n_post_pool_channels: int
        :pooling_type: specifies what type of pooling to use, one of 'max' or
         'average'
        :type pooling_type: str
        :param downsample: if True, use a (2, 2) stride to downsample the
         inputs by a factor of 2
        :type downsample: bool
        """

        super().__init__()

        self.downsample = downsample
        self.n_11_channels = n_11_channels

        if downsample:
            stride = (2, 2)
        else:
            stride = (1, 1)
        
        if n_11_channels:
            self.conv_11 = Conv2d(
                in_channels=n_in_channels, out_channels=n_11_channels,
                kernel_size=(1, 1), stride=stride, bias=False
            )
            self.bn_11 = BatchNorm2d(num_features=n_11_channels)
        self.conv_33_reduce = Conv2d(
            in_channels=n_in_channels, out_channels=n_33_reduce_channels,
            kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.bn_33_reduce = BatchNorm2d(num_features=n_33_reduce_channels)
        self.conv_3333_reduce = Conv2d(
            in_channels=n_in_channels, out_channels=n_3333_reduce_channels,
            kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        self.bn_3333_reduce = BatchNorm2d(num_features=n_3333_reduce_channels)

        self.conv_33 = Conv2d(
            in_channels=n_33_reduce_channels, out_channels=n_33_channels,
            kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False
        )
        self.bn_33 = BatchNorm2d(num_features=n_33_channels)
        self.conv_3333_1 = Conv2d(
            in_channels=n_3333_reduce_channels, out_channels=n_3333_channels,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn_3333_1 = BatchNorm2d(num_features=n_3333_channels)
        self.conv_3333_2 = Conv2d(
            in_channels=n_3333_channels, out_channels=n_3333_channels,
            kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False
        )
        self.bn_3333_2 = BatchNorm2d(num_features=n_3333_channels)
        
        if self.pooling_type == 'max'
            self.pooling = MaxPool2d(kernel_size=(3, 3), stride=(1, 1))
            assert not n_post_pool_channels
        else:
            self.pooling = AvgPool2d(kernel_size=(3, 3), stride=(
            self.conv_post_pooling = Conv2d(
                in_channels=n_in_channels, out_channels=n_post_pool_channels,
                kernel_size=(1, 1), stride=(1, 1), bias=False
            )
            self.bn_post_pool = BatchNorm2d(num_features=n_post_pool_channels)

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
             n_3333_channels + n_post_pool_channels)
        :rtype: torch.Tensor
        """

        if self.n_11_channels:
            layer_11 = self.conv_11(inputs)
            layer_11 = self.bn_11(layer)
            layer_11 = self.relu(layer_11)
        else:
            layer_11 = inputs

        layer_33_reduce = self.conv_33_reduce(inputs)
        layer_33_reduce = self.bn_33_reduce(layer_33_reduce)
        layer_33_reduce = self.relu(layer_33_reduce)
        layer_33 = self.conv_33(layer_33_reduce)
        layer_33 = self.bn_33(layer_33)
        layer_33 = self.relu(layer_33)

        layer_3333_reduce = self.conv_3333_reduce(inputs)
        layer_3333_reduce = self.bn_3333_reduce(layer_3333_reduce)
        layer_3333_reduce = self.relu(layer_3333_reduce)
        layer_3333_1 = self.conv_3333_1(layer_3333_reduce)
        layer_3333_1 = self.bn_3333_1(layer_3333_1)
        layer_3333_1 = self.relu(layer_3333_1)
        layer_3333_2 = self.conv_3333_2(layer_3333_1)
        layer_3333_2 = self.bn_3333_2(layer_3333_2)
        layer_3333_2 = self.relu(layer_3333_2)

        layer_pooled = self.pooling(inputs)
        if downsample:
            layer_post_pooling = layer_pooled
        else:
            layer_post_pooling = self.conv_post_pooling(layer_pooled)
            layer_post_pooling = self.bn_post_pool(layer_post_pooling)
            layer_post_pooling = self.relu(layer_post_pooling)

        outputs = torch.cat(
            (layer_11, layer_33, layer_3333, layer_post_pooling), dim=1
        )
        return outputs


class InceptionNet(Module):
    """Inception network (V2) 
    
    As implemented in 'Batch Normalization: Accelerating Deep 
    Network Training by Reducing Internal Covariate Shift', which is referred
    to as Inception V2 (https://arxiv.org/abs/1502.03167).
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

        n_in_channels = self.config['n_channels']
        n_classes = self.config['n_classes']

        self.conv1 = SeparableConv2d(
            in_channels=n_in_channels, out_channels=64,
            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
            depth_multiplier=8, bias=False
        )
        self.bn1 = BatchNorm2d(num_features=64)
        self.max_pooling1 = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        
        self.conv2 = Conv2d(
            in_channels=64, out_channels=192, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn2 = BatchNorm2d(out_channels=192)
        self.max_pooling2 = MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.inception_3a = Inception(
            n_in_channels=192, n_11_channels=64, n_33_reduce_channels=64,
            n_33_channels=64, n_3333_reduce_channels=64, n_3333_channels=96,
            n_post_pool_channels=32, pooling_type='average', downsample=False
        )
        self.inception_3b = Inception(
            n_in_channels=256, n_11_channels=64, n_33_reduce_channels=64,
            n_33_channels=96, n_3333_reduce_channels=64, n_3333_channels=96,
            n_post_pool_channels=64, pooling_type='average', downsample=False
        )
        self.inception_3c = Inception(
            n_in_channels=320, n_11_channels=0, n_33_reduce_channels=128,
            n_33_channels=160, n_3333_reduce_channels=64, n_3333_channels=96,
            n_post_pool_channels=0, pooling_type='max', downsample=True
        )

        self.inception_4a = Inception(
            n_in_channels=576, n_11_channels=224, n_33_reduce_channels=64,
            n_33_channels=96, n_3333_reduce_channels=96, n_3333_channels=128,
            n_post_pool_channels=128, pooling_type='average', downsample=False
        )
        self.inception_4b = Inception(
            n_in_channels=576, n_11_channels=192, n_33_reduce_channels=96,
            n_33_channels=128, n_3333_reduce_channels=96, n_3333_channels=128,
            n_post_pool_channels=128
        )
        self.inception_4c = Inception(
            n_in_channels=576, n_11_channels=160, n_33_reduce_channels=128,
            n_33_channels=160, n_3333_reduce_channels=128, n_3333_channels=160,
            n_post_pool_channels=128, pooling_type='average', downsample=False
        )
        self.inception_4d = Inception(
            n_in_channels=576, n_11_channels=96, n_33_reduce_channels=128,
            n_33_channels=192, n_3333_reduce_channels=160, n_3333_channels=192,
            n_post_pool_channels=128, pooling_type='average', downsample=False
        )
        self.inception_4e = Inception(
            n_in_channels=576, n_11_channels=0, n_33_reduce_channels=128,
            n_33_channels=192, n_3333_reduce_channels=192, n_3333_channels=256,
            n_post_pool_channels=0, pooling_type='max', downsample=True
        )

        self.inception_5a = Inception(
            n_in_channels=1024, n_11_channels=352, n_33_reduce_channels=192,
            n_33_channels=320, n_3333_reduce_channels=160, n_3333_channels=224,
            n_post_pool_channels=128, pooling_type='average', downsample=False
        )
        self.inception_5b = Inception(
            n_in_channels=1024, n_11_channels=352, n_33_reduce_channels=192,
            n_33_channels=320, n_3333_reduce_channels=192, n_3333_channels=224,
            n_post_pool_channels=128, pooling_type='average', downsample=False
        )

        self.average_pooling = AvgPool2d(kernel_size=7, 7))
        self.dropout = Dropout(0.4)
        self.linear = Linear(
            in_features=1024, out_features=n_classes
        )

        self.auxiliary_classifier1 = AuxiliaryClassifier(
            n_in_channels=576, n_classes=n_classes
        )
        self.auxiliary_classifier2 = AuxiliaryClassifier(
            n_in_channels=576, n_classes=n_classes
        )

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
        layer = self.bn1(layer)
        layer = self.relu(layer)
        layer = self.max_pooling1(layer)
    
        layer = self.conv2(layer)
        layer = self.bn2(layer)
        layer = self.relu(layer)
        layer = self.max_pooling2(layer)

        layer = self.inception_3a(layer)
        layer = self.inception_3b(layer)
        layer = self.inception_3c(layer)

        layer_4a = self.inception_4a(layer)
        layer = self.inception_4b(layer_4a)
        layer = self.inception_4c(layer)
        layer_4d = self.inception_4d(layer)
        layer = self.inception_4e(layer_4d)

        layer = self.inception_5a(layer)
        layer = self.inception_5b(layer)
        layer = self.average_pooling(layer)
        
        layer = layer.reshape(layer.size(0), -1)
        layer = self.dropout(layer)
        outputs = self.linear(layer)

        auxiliary_outputs1 = self.auxiliary_classifier1(layer_4a)
        auxiliary_outputs2 = self.auxiliary_classifier2(layer_4d)

        return auxiliary_outputs1, auxiliary_outputs2, outputs
