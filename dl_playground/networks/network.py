"""Contains the base Network class, from which all networks inherit."""

from utils.generic_utils import validate_config


class Network(object):
    """Abstract base network class"""

    required_config_keys = None

    def __init__(self, network_config):
        """Init

        :param network_config: specifies the configuration for the network
        :type network_config: dict
        """

        if self.required_config_keys:
            validate_config(network_config, self.required_config_keys)

        self.network_config = network_config

    def build(self):
        """Return the inputs and outputs to instantiate a tf.keras.Model object

        :return: inputs and outputs
        :rtype: tuple(tensorflow.Tensor)
        """
