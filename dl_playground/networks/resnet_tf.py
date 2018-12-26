"""ResNet implementation"""

# TODO: Add reference
from tensorflow.keras.layers import (
    Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
)

from utils.generic_utils import validate_config


class ResNet(object):
    """ResNet network"""

    required_config_keys = {'height', 'width', 'n_channels', 'n_classes'}
    
    def __init__(self, network_config):
        """Init

        The `network_config` must contain the following keys:
        - int height: height of the input to the network
        - int width: width of the input to the network
        - int n_channels: number of channels of the input
        - int n_classes: number of classes in the output layer

        :param network_config: specifies the configuration for the network
        :type network_config: dict
        """

        validate_config(network_config, self.required_config_keys)
        self.network_config = network_config

    def _bottleneck_block(self, inputs, n_filters)

        layer = Conv2D(
            filters=n_filters, kernel_size=(1, 1), padding='same'
        )(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), padding='same'
        )(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(
            filters=n_filters * 4, kernel_size=(1, 1), padding='same'
        )(layer)
        layer = BatchNormalization()(layer)

        layer = Add()([layer, inputs])
        layer = Activation('relu')(layer)

    def _identity_shortcut(self, inputs, n_filters):

        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), strides=(2, 2),
            padding='same'
        )(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        
        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), padding='same'
        )(layer)
        layer = BatchNormalization()(layer)

        inputs_shape = inputs.get_shape().as_list()
        layer_shape = layer.get_shape().as_list()

        inputs_padding = get_padding_size(inputs_shape, layer_shape)
        inputs_padding = np.maximum(0, inputs_padding)
        layer_padding = get_padding_size(layer_shape, inputs_shape)
        layer_padding = np.maximum(0, layer_padding)

        layer = ZeroPadding2D(
            padding=layer_padding, data_format='channels_last'
        )(layer)
        # TODO: put note here about why this works
        inputs = ZeroPadding2D(
            padding=inputs_padding, data_format='channels_first'
        )(inputs)

        layer = Add()([layer, inputs])
        layer = Activation('relu')(layer)

        return layer

    def _projection_shorcut(self, inputs, n_filters):

        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), strides=(2, 2),
            padding='same'
        )(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        
        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), padding='same'
        )(layer)
        layer = BatchNormalization()(layer)

        inputs = Conv2D(
            filters=n_filters, kernel_size=(1, 1), strides=(2, 2),
            padding='same',
        )(inputs)

        layer = Add()([layer, inputs])
        layer = Activation('relu')(layer)

        return layer
        
    def _residual_block(self, inputs, n_filters):

        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), padding='same'
        )(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), padding='same'
        )(layer)
        layer = BatchNormalization()(layer)

        layer = Add()([layer, inputs])
        layer = Activation('relu')(layer)

        return layer

    def build(self):
        """Return the inputs and outputs to instantiate a tf.keras.Model object

        :return: inputs and outputs
        :rtype: tuple(tensorflow.Tensor)
        """

        heigth = self.config['height']
        width = self.config['width']
        n_channels = self.config['n_channels']
        n_classes = self.config['n_classes']

        n_initial_filters = self.config['n_initial_filters']
        initial_conv_kernel_size = self.config['initial_conv_kernel_size']
        n_blocks_per_stage = self.config['n_blocks_per_stage']

        if self.config['shorcut_type'] == 'identity':
            shortcut_fn = self._identity_shortcut
        else:
            shortcut_fn = self._projection_shorcut
        if self.config['residual_block_type'] == 'basic':
            residual_block_fn = self._residual_block
        else:
            residual_block_fn = self._bottleneck_block

        inputs = Input(shape=(height, width, n_channels), name='image')

        n_filters = n_initial_filters
        layer = Conv2D(
            filters=n_filters, kernel_size=initial_conv_kernel_size,
            strides=(2, 2), padding='same',
        )(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        # TODO: check this max pooling strides of 1
        layer = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(layer)

        for idx_stage, n_blocks in enumerate(n_blocks_per_stage):
            n_filters *= 2
            if idx_stage != 0:
                n_blocks -= 1
                layer = shorcut_fn(layer, n_filters=n_filters)

            for _ in range(n_blocks):
                layer = residual_block_fn(layer, n_filters=n_filters)

        layer = GlobalAveragePooling2D()(layer)
        outputs = Dense(
            units=n_classes, activation='softmax', name='label'
        )(layer)

        return inputs, outputs
