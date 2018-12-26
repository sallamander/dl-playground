
# Source:
# https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua
# https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras

class HourglassNet(object):
    """"""

    def __init__(self, config):
        
        validate_config(config, self.requirede_config_keys)
        self.network_config(network_config)

    # TODO: rework the bottleneck network from the residual network version of
    # tensorflow (although this one is BN relu conv)
    def _bottleneck_block(self, inputs, n_filters):

        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(
            filters=n_filters, kernel_size=(1, 1), padding='same',
        )(layer)

        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), padding='same',
        )(layer)

        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(
            filters=n_filters * 2, kernel_size=(1, 1), padding='same',
        )(layer)

    def _hourglass_module(self, layer, n_filters):

        unpooled_layers = []
        # pooled_layers = []
        for _ in range(4):
            layer = self._bottleneck_block(layer, n_filters)
            unpooled_layers.append(layer)

            layer = MaxPooling2D(pool_size=(2, 2))(layer)
            # pooled_layers.append(layers)

        layer = self._bottleneck_block(layer, n_filters)
        layer = self._bottleneck_block(layer, n_filters)

        unpooled_layers = unpooled_layers[::-1]
        for idx_upsampling in range(4):
            # TODO: Will need tensorflow 1.13 for this
            layer = Upsampling2D(interpolation='nearest')(layer)
            layer = Add()[layer, unpooled_layers[idx_upsampling]]
            
            layer = self._bottleneck_block(layer, n_filters)

        return layer


    def build(self):

        height = self.config['height']
        width = self.config['width']
        n_channels = self.config['n_channels']
        n_classes = self.config['n_classes']

        n_hourglass_modules = self.config['n_hourglass_modules']

        inputs = Input(shape=(height, width, n_channels), name='image')

        layer = Conv2D(
            filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same'
        )(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        layer = self._bottleneck_block(layer, n_filters=128)
        layer = MaxPooling2D(pool_size=(2, 2))
        layer = self._bottleneck_block(layer, n_filters=128)
        layer = self._bottleneck_block(layer, n_filters=256)

        hourglass_input = layer
        
        outputs = []
        for idx_module in range(n_hourglass_modules):
            layer = self._hourglass_module(hourglass_input, n_filters=256)
            layer = Conv2D(
                filters=256, kernel_size=(1, 1), strides=(1, 1),
                padding='valid'
            )(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)

            hourglass_output = Conv2D(
                filters=n_classes, kernel_size=(1, 1), strides=(1, 1),
                padding='valid'
            )(layer)
            outputs.append(hourglass_output)

            if idx_module < n_hourglass_modules:
                layer = Conv2D(
                    filters=256, kernel_size=(1, 1), strides=(1, 1),
                    padding='valid'
                )(layer)
                higher_dimensional_output = Conv2D(
                    filters=256, kernel_size=(1, 1), strides=(1, 1),
                    padding='valid'
                )(hourglass_output)
                hourglass_input = Add()(
                    [hourglass_input, layer, higher_dimensional_output]
                )

        return inputs, outputs
