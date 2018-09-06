"""AlexNet implementation

Reference paper (using bitly link to save line length): https://bit.ly/2v4Aihl

The main difference between this implementation and the paper is that it does
not use any parallelization across GPUs.
"""

from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
)


class AlexNet(object):
    """AlexNet model"""

    def __init__(self, network_config):
        """Init

        network_config must contain the following keys:
        - int or float height: height of the input to the network
        - int or float width: width of the input to the network
        - int or float n_channels: number of channels of the input
        - int or float n_classes: number of classes in the output layer

        :param network_config: specifies the configuration for the network
        :type network_config: dict
        """

        self._validate_config(network_config)

        self.height = network_config['height']
        self.width = network_config['width']
        self.n_channels = network_config['n_channels']
        self.n_classes = network_config['n_classes']

    @staticmethod
    def _validate_config(network_config):
        """Validate that the necessary keys are in the network_config

        This raises a KeyError if there are required keys that are missing, and
        otherwise does nothing.

        :param network_config: specifies the configuration for the network
        :type network_config: dict
        """

        required_keys = {'height', 'width', 'n_channels', 'n_classes'}
        missing_keys = required_keys - set(network_config)

        if missing_keys:
            msg = (
                '{} keys are missing from the network_config, but are '
                'required in order to construct the model.'
            ).format(missing_keys)
            raise KeyError(msg)

    def build(self):
        """Return the inputs and outputs to instantiate a tf.keras.Model object

        :return: inputs and outputs
        :rtype: tuple(tf.Tensor)
        """

        inputs = Input(shape=(self.height, self.width, self.n_channels))

        # === convolutional block 1 === #
        layer = Conv2D(
            filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same',
            activation='relu'
        )(inputs)
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

        # === convolutional block 2 === #
        layer = Conv2D(
            filters=256, kernel_size=(5, 5), padding='same', activation='relu'
        )(layer)
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

        # === convolutional blocks 3, 4, 5 === #
        layer = Conv2D(
            filters=384, kernel_size=(3, 3), padding='same', activation='relu'
        )(layer)
        layer = Conv2D(
            filters=384, kernel_size=(3, 3), padding='same', activation='relu'
        )(layer)
        layer = Conv2D(
            filters=256, kernel_size=(3, 3), padding='same', activation='relu'
        )(layer)
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

        # === dense layers (2) === #
        layer = Flatten()(layer)
        layer = Dense(units=4096, activation='relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(units=4096, activation='relu')(layer)
        layer = Dropout(0.5)(layer)

        # === output layer === #
        outputs = Dense(units=self.n_classes, activation='softmax')(layer)

        return inputs, outputs
