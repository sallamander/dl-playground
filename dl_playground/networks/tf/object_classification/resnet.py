"""ResNet implementation written with TensorFlow

Reference Papers:
    - 'Original' (V1): 'Deep Residual Learning for Image Recognition'
        https://arxiv.org/abs/1512.03385
    - 'Pre-Activation' (V2): 'Identity Mappings in Deep Residual Networks'
        https://arxiv.org/abs/1603.05027

Main Reference Implementations:
    - 'Original' (V1): https://github.com/KaimingHe/deep-residual-networks/blob
        /master/prototxt/ResNet-50-deploy.prototxt
    - 'Pre-Activation' (V2): https://github.com/KaimingHe/resnet-1k-layers/blob
        /master/resnet-pre-act.lua

Other Reference Implementations:
    - 'Original' (V1) and 'Pre-Activation' (V2):
        - https://github.com/taehoonlee/tensornets/blob/master/tensornets
            /resnets.py
    - 'Original' (V2):
        - https://github.com/keras-team/keras-applications/blob/master
            /keras_applications/resnet50.py
    - 'Pre-Activation' (V2):
        - https://github.com/keras-team/keras-applications/blob/master
            /keras_applications/resnet_common.py

This implementation follows the "Main Reference Implementation" as closely as
possible. The "Other Reference Implementations" were used as a reference, and
sometimes appear to have slight differences from the main implementation.
"""

import math

from tensorflow.keras.layers import (
    Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense,
    Dropout, Flatten, Input, MaxPooling2D
)

from trainet.utils.generic_utils import validate_config


class ResNet():
    """ResNet network

    The version key in the `config` passed to the init should specify one of
    the following:
        - 'original': 'Deep Residual Learning for Image Recognition'
            (https://arxiv.org/abs/1512.03385)
        - 'preactivation': 'Identity Mappings in Deep Residual Networks'
            (https://arxiv.org/abs/1603.05027)
    """

    required_config_keys = {
        'height', 'width', 'n_channels', 'n_classes', 'n_initial_filters',
        'n_blocks_per_stage', 'version'
    }

    def __init__(self, config):
        """Init

        The `config` must contain the following keys:
        - int height: height of the input to the network
        - int width: width of the input to the network
        - int n_channels: number of channels of the input
        - int n_classes: number of classes in the output layer
        - int n_initial_filters: number of filters in the initial convolution
          as well as the first residual block; subsequent residual blocks will
          have N times the filters as the previous block
        - list[int] n_blocks_per_stage: iterable holding the number of residual
          blocks to use per residual stage; the length of n_blocks_per_stage
          defines the number of residual stages in the network
        - str version: one of 'original' or 'preactivation'

        :param config: specifies the configuration for the network
        :type config: dict
        """

        validate_config(config, self.required_config_keys)
        self.config = config

        self.version = config['version']
        assert self.version in {'original', 'preactivation'}

    def _bottleneck_block_v1(self, inputs, n_mid_filters, n_out_filters):
        """Return the output of a bottleneck block (V1) applied to `inputs`

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_in_channels)
        :type inputs: tensorflow.Tensor
        :param n_mid_filters: number of filters to use in the convolutions
         applied to the inputs (minus the last convolution)
        :type n_mid_filters: int
        :param n_out_filters: number of filters to use in the final convolution
         in the block
        :type int
        :return: outputs of a bottleneck block, of shape
         (batch_size, height, width, n_out_filters)
        :rtype: tensorflow.Tensor
        """

        # === 1x1 conv === #
        layer = Conv2D(
            filters=n_mid_filters, kernel_size=(1, 1),
            padding='same', use_bias=False
        )(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        # === 3x3 conv === #
        layer = Conv2D(
            filters=n_mid_filters, kernel_size=(3, 3),
            padding='same', use_bias=False
        )(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        # === 1x1 conv === #
        layer = Conv2D(
            filters=n_out_filters, kernel_size=(1, 1),
            padding='same', use_bias=False
        )(layer)
        layer = BatchNormalization()(layer)

        layer = Add()([layer, inputs])
        layer = Activation('relu')(layer)
        return layer

    def _bottleneck_block_v2(self, inputs, n_mid_filters, n_out_filters):
        """Return the output of a bottleneck block (V2) applied to `inputs`

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_in_channels)
        :type inputs: tensorflow.Tensor
        :param n_mid_filters: number of filters to use in the convolutions
         applied to the inputs (minus the last convolution)
        :type n_mid_filters: int
        :param n_out_filters: number of filters to use in the final convolution
         in the block
        :type int
        :return: outputs of a bottleneck bloc, of shape
         (batch_size, height, width, n_out_filters)
        :rtype: tensorflow.Tensor
        """

        # === 1x1 conv === #
        layer = BatchNormalization()(inputs)
        layer = Activation('relu')(layer)
        layer = Conv2D(
            filters=n_mid_filters, kernel_size=(1, 1),
            padding='same', use_bias=False
        )(layer)

        # === 3x3 conv === #
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(
            filters=n_mid_filters, kernel_size=(3, 3),
            padding='same', use_bias=False
        )(layer)

        # === 1x1 conv === #
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(
            filters=n_out_filters, kernel_size=(1, 1),
            padding='same', use_bias=False
        )(layer)

        layer = Add()([layer, inputs])
        return layer

    def _projection_shortcut_v1(self, inputs, n_mid_filters, n_out_filters,
                                stride):
        """Return the output of a projection shortcut (V1) applied to `inputs`

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_in_channels)
        :type inputs: tensorflow.Tensor
        :param n_mid_filters: number of filters to use in the convolutions
         applied to the inputs (minus the last convolution)
        :type n_mid_filters: int
        :param n_out_filters: number of filters to use in the final convolution
         in the block
        :type int
        :param stride: stride to use in the *first* convolution applied to
         `inputs` in the projection shortcut; controls the level of downsampling
         that is performed
        :type stride: tuple(int)
        :return: outputs of a projection shortcut, of shape
         (batch_size, height / 2, width / 2, n_out_filters)
        :rtype: tensorflow.Tensor
        """

        # === 1x1 conv === #
        layer = Conv2D(
            filters=n_mid_filters, kernel_size=(1, 1), strides=stride,
            use_bias=False, padding='same'
        )(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        # === 3x3 conv === #
        layer = Conv2D(
            filters=n_mid_filters, kernel_size=(3, 3),
            padding='same', use_bias=False
        )(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        # === 1x1 conv === #
        layer = Conv2D(
            filters=n_out_filters, kernel_size=(1, 1),
            padding='same', use_bias=False
        )(layer)
        layer = BatchNormalization()(layer)

        # === projection conv === #
        shortcut = Conv2D(
            filters=n_out_filters, kernel_size=(1, 1), strides=stride,
            padding='same', use_bias=False
        )(inputs)
        shortcut = BatchNormalization()(shortcut)

        layer = Add()([layer, shortcut])
        layer = Activation('relu')(layer)
        return layer

    def _projection_shortcut_v2(self, inputs, n_mid_filters, n_out_filters,
                                stride):
        """Return the output of a projection shortcut (V2) applied to `inputs`

        :param inputs: batch of input images, of shape
         (batch_size, height, width, n_in_channels)
        :type inputs: tensorflow.Tensor
        :param n_mid_filters: number of filters to use in the convolutions
         applied to the inputs (minus the last convolution)
        :type n_mid_filters: int
        :param n_out_filters: number of filters to use in the final convolution
         in the block
        :type int
        :param stride: stride to use in the *first* convolution applied to
         `inputs` in the projection shortcut; controls the level of downsampling
         that is performed
        :type stride: tuple(int)
        :return: outputs of a projection shortcut, of shape
         (batch_size, height / 2, width / 2, n_out_filters)
        :rtype: tensorflow.Tensor
        """

        preactivation = BatchNormalization()(inputs)
        preactivation = Activation('relu')(preactivation)

        # === 1x1 conv === #
        layer = Conv2D(
            filters=n_mid_filters, kernel_size=(1, 1), strides=stride,
            use_bias=False, padding='same'
        )(preactivation)

        # === 3x3 conv === #
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(
            filters=n_mid_filters, kernel_size=(3, 3),
            padding='same', use_bias=False
        )(layer)

        # === 1x1 conv === #
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(
            filters=n_out_filters, kernel_size=(1, 1),
            padding='same', use_bias=False
        )(layer)

        # === projection conv === #
        shortcut = Conv2D(
            filters=n_out_filters, kernel_size=(1, 1), strides=stride,
            padding='same', use_bias=False
        )(preactivation)

        layer = Add()([layer, shortcut])
        return layer

    def build(self):
        """Return the inputs and outputs to instantiate a tf.keras.Model object

        :return: inputs and outputs
        :rtype: tuple(tensorflow.Tensor)
        """

        height = self.config['height']
        width = self.config['width']
        n_channels = self.config['n_channels']
        n_classes = self.config['n_classes']

        n_initial_filters = self.config['n_initial_filters']
        n_blocks_per_stage = self.config['n_blocks_per_stage']

        if self.version == 'original':
            bottleneck = self._bottleneck_block_v1
            projection_shortcut = self._projection_shortcut_v1
        else:
            bottleneck = self._bottleneck_block_v2
            projection_shortcut = self._projection_shortcut_v2

        inputs = Input(shape=(height, width, n_channels), name='image')

        n_filters = n_initial_filters
        layer = Conv2D(
            filters=n_filters, kernel_size=(7, 7),
            strides=(2, 2), padding='same',
        )(inputs)

        if self.version == 'original':
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

        n_mid_filters = n_filters
        for idx_stage, n_blocks in enumerate(n_blocks_per_stage):
            if idx_stage == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)

            n_mid_filters = (n_initial_filters * (2 ** idx_stage))
            n_out_filters = (
                n_initial_filters * (2 ** idx_stage) * 4
            )

            layer = projection_shortcut(
                layer, n_mid_filters, n_out_filters, stride
            )

            for _ in range(n_blocks):
                layer = bottleneck(layer, n_mid_filters, n_out_filters)

        if self.version == 'preactivation':
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)

        layer = AveragePooling2D(pool_size=(7, 7))(layer)
        layer = Flatten()(layer)
        outputs = Dense(
            units=n_classes, activation='softmax', name='label'
        )(layer)

        return inputs, outputs
