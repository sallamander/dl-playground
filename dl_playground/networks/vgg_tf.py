"""VGG[16 | 19] implementations"""

# TODO: Add reference
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
)

from utils.generic_utils import validate_config


class VGG(object):
    """VGG network

    Both VGG16 and VGG19 are available using this class, with the option
    controlled via the `n_layers` parameter passed to the `__init__`.
    """

    required_config_keys = {'height', 'width', 'n_channels', 'n_classes'}
    
    def __init__(self, network_config, n_layers=16):
        """Init

        The `network_config` must contain the following keys:
        - int height: height of the input to the network
        - int width: width of the input to the network
        - int n_channels: number of channels of the input
        - int n_classes: number of classes in the output layer

        :param network_config: specifies the configuration for the network
        :type network_config: dict
        :param n_layers: number of layers, either 16 for VGG16 or 19 for VGG19
        :type n_layers: int
        """

        validate_config(network_config, self.required_config_keys)
        if n_layers not in {16, 19}:
            msg = (
                '`n_layers` must be one of {16, 19} to specify VGG16 or VGG19.'
            )
            raise ValueError(msg)

        self.network_config = network_config
        self.n_layers = n_layers

    def _add_conv_stack(inputs, n_conv_layers, n_filters):
        """Apply a stack of convolutional layers to the `inputs`

        :param inputs: tensor to apply the convolutional layers to
        :type inputs: tensorflow.Tensor
        :param n_conv_layers: number of convolutional layers to apply to the
         `inputs`
        :type n_conv_layers: int
        :param n_filters: number of filters to use in each of the convolutional
         layers
        :type n_filters: int
        """

        layer = inputs

        for _ in range(n_conv_layers):
            layer = Conv2D(
                filters=n_filters, kernel_size=(3, 3),
                padding='same', activation='relu'
            )(layer)
        return layer

    def build(self):
        """Return the inputs and outputs to instantiate a tf.keras.Model object

        :return: inputs and outputs
        :rtype: tuple(tensorflow.Tensor)
        """

        inputs = Input(shape=(height, width, n_channels), name='image')

        # === convolutional stacks 1 and 2 === #
        layer = self._add_conv_stack(inputs, n_conv_layers=2, n_filters=64)
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
        layer = self._add_conv_stack(layer, n_conv_layers=2, n_filters=128)
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

        # === convolutional stacks 3, 4, and 5 === #
        n_conv_layers = 3 if self.n_layers == 16 else 4
        layer = self._add_conv_stack(
            layer, n_conv_layers=n_conv_layers, n_filters=256
        )
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
        layer = self._add_conv_stack(
            layer, n_conv_layers=n_conv_layers, n_filters=512
        )
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
        layer = self._add_conv_stack(
            layer, n_conv_layers=n_conv_layers, n_filters=512
        )
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

        # === dense layers (2) === #
        layer = Flatten()(layer)
        layer = Dense(units=4096, activation='relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(units=4096, activation='relu')(layer)
        layer = Dropout(0.5)(layer)

        # === output layer === #
        outputs = Dense(
            units=n_classes, activation='softmax', name='label'
        )(layer)

        return inputs, outputs
