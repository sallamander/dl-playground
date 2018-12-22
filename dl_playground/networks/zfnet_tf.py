"""ZFNet implementation written with tensorflow.keras"""

# TODO: Add paper reference

from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
)

from utils.generic_utils import validate_config


class ZFNet(object):
    """AlexNet model"""

    required_config_keys = {'height', 'width', 'n_channels', 'n_classes'}

    def __init__(self, network_config):
        """Init

        `network_config` must contain the following keys:
        - int height: height of the input to the network
        - int width: width of the input to the network
        - int n_channels: number of channels of the input
        - int n_classes: number of classes in the output layer

        :param network_config: specifies the configuration for the network
        :type network_config: dict
        """

        validate_config(network_config, self.required_config_keys)
        self.network_config = network_config

    def forward(self):
        """Return the inputs and outputs representing a forward pass of ZFNet

        The returned inputs and outputs can be placed right into a
        `tf.keras.Model` object as the `inputs` and `outputs` arguments.

        :return: inputs of shape (batch_size, height, width, n_channels)
         and outputs of shape (batch_size, n_classes)
        :rtype: tuple(tensorflow.Tensor)
        """

        height = self.network_config['height']
        width = self.network_config['width']
        n_channels = self.network_config['n_channels']
        n_classes = self.network_config['n_classes']

        inputs = Input(shape=(height, width, n_channels), name='image')

        # === convolutional block 1 === #
        layer = Conv2D(
            filters=96, kernel_size=(7, 7), strides=(2, 2), padding='valid',
            activation='relu'
        )(inputs)
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

        # === convolutional block 2 === #
        layer = Conv2D(
            filters=256, kernel_size=(5, 5), padding='valid', activation='relu'
        )(layer)
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

        # === convolutional blocks 3, 4, 5 === #
        layer = Conv2D(
            filters=384, kernel_size=(3, 3), padding='valid', activation='relu'
        )(layer)
        layer = Conv2D(
            filters=384, kernel_size=(3, 3), padding='valid', activation='relu'
        )(layer)
        layer = Conv2D(
            filters=256, kernel_size=(3, 3), padding='valid', activation='relu'
        )(layer)
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

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
