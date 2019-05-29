"""Region Proposal Network for object detection"""


class RegionProposalNetwork(object):
    """Region Proposal Network (RPN)"""

    required_config_keys = {'n_filters', 'n_aspect_ratios', 'n_anchor_scales'}

    def __init__(self, config):
        """Init

        `config` must contain the following keys:
        - int height: height of the input to the network
        - int width: width of the input to the network
        - int n_channels: number of channels of the input

        :param config: specifies the configuration for the network
        :type config: dict
        """

        validate_config(config, self.required_config_keys)
        self.config = config

    # TODO: build the RPN on top of `inputs`
    def forward(self, inputs):
        """Return the inputs/outputs representing a forward pass of RPN

        :return: inputs of shape (batch_size, height, width, n_channels) and
         outputs of shape
         (batch_size, feature_map_height, feature_map_width, n_channels)
        :rtype: tuple(tensorflow.Tensor)
        """

        n_filters = self.config['n_filters']
        n_anchors = (
            len(self.config['anchor_scales']) *
            len(self.config['anchor_aspect_ratios'])
        )

        layer = Conv2D(
            filters=n_filters, kernel_size=(3, 3), padding='same',
            activation='relu'
        )(inputs)

        classification_outputs = Conv2D(
            filters=(2 * n_anchors), kernel_size=(1, 1), padding='same',
            activation='sigmoid', kernel_initializer=normal(stddev=0.01)
        )
        regression_outputs = Conv2D(
            filters=(4 * n_anchors), kernel_size=(1, 1), padding='same',
            activation='linear', kernel_initializer=normal(stddev=0.01)
        )(layer)

        return classification_outputs, regression_outputs
