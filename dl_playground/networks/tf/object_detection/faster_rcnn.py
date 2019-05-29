"""Faster RCNN implementation written with tensorflow.keras

Reference paper: https://arxiv.org/abs/1506.01497
"""

from tensorflow.keras.layers import Conv2D

from utils.generic_utils import import_object, validate_config


class FasterRCNN(object):
    """Faster RCNN model"""

    required_config_keys = {
        'region_proposal_network'
    }

    def __init__(self, config):
        """Init

        `config` must contain the following keys:
        - dict region_proposal_network: acts as the `config` to pass to a
          network that will act as the Region Proposal Network

        :param config: specifies the configuration for the network
        :type config: dict
        """

        validate_config(config, self.required_keys)
        self.config = config

    # TODO: Fill in this docstring
    def _get_backbone(self):
        """Return the backbone network holding the shared convs"""

        BackboneNetwork = import_object(self.config['backbone']['importpath'])
        backbone_config = self.config['backbone']['config']

        backbone = BackboneNetwork(**backbone_config)
        return backbone

    # TODO: Fill out this docstring
    def _get_region_proposals(self, shared_conv_features):
        """"""

        RegionProposalNetwork = (
            import_object(self.config['region_proposal_network']['importpath'])
        )
        region_proposal_network_config = (
            self.config['region_proposal_network']['config']
        )

        region_proposal_network = (
            RegionProposalNetwork(**region_proposal_network_config)
        )
        classification_outputs, regression_outputs = (
            region_proposal_network.build(shared_conv_features)
        )
        return classification_outputs, regression_outputs



    # TODO: Fill in the output shape
    def forward(self):
        """Return the inputs/outputs representing a forward pass of FasterRCNN

        :return: inputs of shape (batch_size, height, width, n_channels)
         and outputs of shape (TODO)
        :rtype: tuple(tensorflow.Tensor)
        """
        
        backbone = self._get_backbone()
        inputs, shared_conv_features = backbone.as_feature_extractor()
        rpn_anchor_classifications, rpn_anchor_regressions = (
            self._get_region_proposals(shared_conv_features)
        )
