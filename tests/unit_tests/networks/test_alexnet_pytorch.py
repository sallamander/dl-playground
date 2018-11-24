"""Unit tests for networks.alexnet_pytorch"""

from contextlib import ExitStack
from unittest.mock import patch, MagicMock

import pytest
import torch

import networks.alexnet_pytorch

AlexNet = networks.alexnet_pytorch.AlexNet


class TestAlexNet(object):
    """Tests for AlexNet"""

    @pytest.fixture(scope='class')
    def network_config(self):
        """network_config object fixture

        :return: network_config to be used with AlexNet
        :rtype: dict
        """

        return {'n_channels': 3, 'n_classes': 1000}

    @pytest.fixture(scope='class')
    def layers_dict(self):
        """layers_dict object fixture

        This holds as keys strings denoting the common name of one of the
        layers in an AlexNet model (e.g. 'convolutional') and as values a
        wrapped `unittest.mock._patch` object to allow for checking the
        `call_count` of each of these layers.
        """

        layers_dict = {}
        layers_dict['convolutional'] = {
            'expected_count': 5,
            'wrapped_layer': patch.object(
                networks.alexnet_pytorch, 'Conv2d',
                wraps=networks.alexnet_pytorch.Conv2d
            )
        }
        layers_dict['relu'] = {
            'expected_count': 7,
            'wrapped_layer': patch.object(
                networks.alexnet_pytorch, 'ReLU',
                wraps=networks.alexnet_pytorch.ReLU
            )
        }
        layers_dict['max_pooling'] = {
            'expected_count': 3,
            'wrapped_layer': patch.object(
                networks.alexnet_pytorch, 'MaxPool2d',
                wraps=networks.alexnet_pytorch.MaxPool2d
            )
        }
        layers_dict['linear'] = {
            'expected_count': 3,
            'wrapped_layer': patch.object(
                networks.alexnet_pytorch, 'Linear',
                wraps=networks.alexnet_pytorch.Linear
            )
        }
        layers_dict['dropout'] = {
            'expected_count': 2,
            'wrapped_layer': patch.object(
                networks.alexnet_pytorch, 'Dropout',
                wraps=networks.alexnet_pytorch.Dropout
            )
        }

        return layers_dict

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

        mock_validate_config = MagicMock()
        monkeypatch.setattr(
            'networks.alexnet_pytorch.validate_config', mock_validate_config
        )
        alexnet = AlexNet(network_config)

        assert alexnet.required_config_keys == {'n_channels', 'n_classes'}
        assert id(network_config) == id(alexnet.network_config)
        assert mock_validate_config.call_count == 1
        mock_validate_config.assert_called_with(
            network_config, AlexNet.required_config_keys
        )

    def test_forward(self, network_config, layers_dict):
        """Test forward method

        :param network_config: network_config object fixture
        :type network_config: dict
        :param layers_dict: layers_dict object fixture
        :type layers_dict: dict
        """

        alexnet = MagicMock()
        alexnet.network_config = network_config
        alexnet.forward = AlexNet.forward

        inputs = torch.randn((1, 227, 227, 3))

        wrapped_layer_contexts = []
        with ExitStack() as stack:
            for _, layer_dict in layers_dict.items():
                wrapped_layer_context = (
                    stack.enter_context(layer_dict['wrapped_layer'])
                )
                wrapped_layer_contexts.append(
                    (wrapped_layer_context, layer_dict['expected_count'])
                )
            outputs = alexnet.forward(self=alexnet, inputs=inputs)
            assert outputs.shape == (1, 1000)

            it = wrapped_layer_contexts
            for wrapped_layer_context, expected_call_count in it:
                assert (
                    wrapped_layer_context.call_count ==
                    expected_call_count
                )
