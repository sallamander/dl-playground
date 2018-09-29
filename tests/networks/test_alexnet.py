"""Tests for networks.alexnet.py"""

import pytest

import numpy as np
import tensorflow as tf

from networks.alexnet import AlexNet


class TestAlexNet(object):
    """Tests for AlexNet"""

    @pytest.fixture(scope='class')
    def network_config(self):
        """network_config object fixture

        :return: network_config to be used with AlexNet
        :rtype: dict
        """

        return {
            'height': 227, 'width': 227, 'n_channels': 3, 'n_classes': 1000
        }

    def test_init(self, network_config):
        """Test __init__ method

        This tests two things:
        - All attributes are set correctly in the __init__
        - A KeyError is raised if 'height', 'width', 'n_channels', or
          'n_classes' is not present in the `network_config`

        :param network_config: network_config object fixture
        :type network_config: dict
        """

        # === test all attributes are set correctly === #
        alexnet = AlexNet(network_config)

        assert alexnet.height == 227
        assert alexnet.width == 227
        assert alexnet.n_channels == 3
        assert alexnet.n_classes == 1000

        # === test `network_config` === #
        for network_key in network_config:
            network_config_copy = network_config.copy()
            del network_config_copy[network_key]

            with pytest.raises(KeyError):
                AlexNet(network_config_copy)

    def test_build(self, network_config):
        """Test build method

        This tests a couple of things:
        - The input and output layers are of the right type and shape
        - The correct number of layers exist in the network, and of the
          expected types
        """

        alexnet = AlexNet(network_config)
        inputs, outputs = alexnet.build()

        assert isinstance(inputs, tf.Tensor)
        assert isinstance(outputs, tf.Tensor)
        assert np.allclose(
            inputs.get_shape().as_list()[1:], (227, 227, 3)
        )
        assert np.allclose(
            outputs.get_shape().as_list()[1:], (1000)
        )

        model = tf.keras.Model(inputs, outputs)
        layers_dict = {
            'convolutional': {'expected_count': 5, 'count': 0},
            'dense': {'expected_count': 3, 'count': 0},
            'dropout': {'expected_count': 2, 'count': 0},
            'flatten': {'expected_count': 1, 'count': 0},
            'input': {'expected_count': 1, 'count': 0},
            'max_pooling': {'expected_count': 3, 'count': 0},
        }

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_type = 'convolutional'
            elif isinstance(layer, tf.keras.layers.Dense):
                layer_type = 'dense'
            elif isinstance(layer, tf.keras.layers.Dropout):
                layer_type = 'dropout'
            elif isinstance(layer, tf.keras.layers.Flatten):
                layer_type = 'flatten'
            elif isinstance(layer, tf.keras.layers.InputLayer):
                layer_type = 'input'
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer_type = 'max_pooling'

            layers_dict[layer_type]['count'] += 1

        for layer_type in layers_dict:
            expected_count = layers_dict[layer_type]['expected_count']
            count = layers_dict[layer_type]['count']
            assert expected_count == count
