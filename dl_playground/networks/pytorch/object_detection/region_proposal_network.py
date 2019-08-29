"""Region Proposal Network For Object Detection

Reference paper:
    - 'Faster R-CNN: Towards Real-Time Object Detection with Region '
      'Proposal Networks'
          https://arxiv.org/abs/1506.01497
"""

from torch.nn import Conv2d, Module, ReLU


class RegionProposalNetwork(Module):
    """Region Proposal Network (RPN)"""

    required_config_keys = {
        'n_in_channels', 'n_out_channels',
        'n_aspect_ratios', 'n_anchor_scales'
    }

    def __init__(self, config):
        """Init

        `config` must contain the following keys:
        - int n_in_channels: number of filters in the inputs to the region
          proposal network
        - int n_out_channels: number of filters in the 3x3 convolutional layer
          applied to the inputs
        - int n_aspect_ratios: number of aspect ratios used for the anchors
        - int n_anchor_scales: number of scales used for the anchors

        :param config: specifies the configuration for the network
		:type config: dict
        """
        
        validate_config(config, self.required_keys)
        self.config = config
        
        n_in_channels = self.config['n_in_channels']
        n_out_channels = self.config['n_out_channels']
        n_anchors = (
            len(self.config['anchor_scales']) *
            len(self.config['anchor_aspect_ratios'])
        )

        self.conv = Conv2d(
            in_channels=n_in_channels, out_channels=n_out_channels,
            kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)
        )
        self.classification_conv = Conv2d(
            in_channels=n_out_channels, out_channels=(n_anchors * 2),
            kernel_size=(1, 1), stride=(1, 1)
        )
        self.regression_conv = Conv2d(
            in_channels=n_out_channels, out_channels=(n_anchors * 4),
            kernel_size=(1, 1), stride=(1, 1)
        )

        self.relu = ReLU()


    def forward(self, inputs):
        """Return the output of a forward pass of the RPN

        :param inputs: batch of input images, of shape
         (batch_size, n_in_channels, height, width)
		:type inputs: torch.Tensor
        :return: classification and regression outputs of a forward pass of
         RPN, where the classification outputs are of shape
          (batch_size, n_anchors * 2, height, width) and the regression outputs
          are of shape (batch_size, n_anchors * 4, height, width)
		:rtype: tuple(torch.Tensor)
        """

        layer = self.conv(inputs)
        layer = self.relu(layer)
        
        classification_outputs = self.classification_conv(layer)
        regression_outputs = self.regression_conv(layer)

        return classification_outputs, regression_outputs
