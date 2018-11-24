"""Unit tests for networks.alexnet_tf"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from networks.alexnet_tf import AlexNet, Network


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

    def test_init(self, network_config, monkeypatch):
        """Test __init__ method

        This tests two things:
        - All attributes are set correctly in the __init__
        - The `required_config_keys` attribute holds the expected values

        :param network_config: network_config object fixture
        :type network_config: dict
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        assert AlexNet.required_config_keys == {
            'height', 'width', 'n_channels', 'n_classes'
        }

        def mock_init(self, network_config):
            """Mock __init__

            The `__init__` only needs to set `self.network_config` equal to the
            input `network_config.`
            """
            self.network_config = network_config
        monkeypatch.setattr(Network, '__init__', mock_init)

        alexnet = AlexNet(network_config)
        for key, value in network_config.items():
            assert alexnet.network_config[key] == value

    def test_build(self, network_config):
        """Test build method

        This tests a couple of things:
        - The input and output layers are of the right type and shape
        - The correct number of layers exist in the network, and of the
          expected types
        """

        alexnet = MagicMock()
        alexnet.build = AlexNet.build
        alexnet.network_config = network_config
        inputs, outputs = alexnet.build(self=alexnet)

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
