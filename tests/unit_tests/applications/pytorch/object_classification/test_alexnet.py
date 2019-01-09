"""Unit tests for networks.pytorch.object_classification.alexnet"""

from contextlib import ExitStack
from unittest.mock import patch, MagicMock

import pytest
import torch

import networks.pytorch.object_classification.alexnet as pytorch_alexnet

AlexNet = pytorch_alexnet.AlexNet


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
        layers_dict['relu'] = {
            'expected_count': 7,
            'wrapped_layer': patch.object(
                pytorch_alexnet, 'ReLU',
                wraps=pytorch_alexnet.ReLU
            )
        }
        layers_dict['max_pooling'] = {
            'expected_count': 3,
            'wrapped_layer': patch.object(
                pytorch_alexnet, 'MaxPool2d',
                wraps=pytorch_alexnet.MaxPool2d
            )
        }
        layers_dict['dropout'] = {
            'expected_count': 2,
            'wrapped_layer': patch.object(
                pytorch_alexnet, 'Dropout',
                wraps=pytorch_alexnet.Dropout
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
            ('networks.pytorch.object_classification.alexnet'
             '.validate_config'),
            mock_validate_config
        )
        mock_set_layers = MagicMock()
        monkeypatch.setattr(
            ('networks.pytorch.object_classification.alexnet'
             '.AlexNet._set_layers'),
            mock_set_layers
        )
        alexnet = AlexNet(network_config)

        assert alexnet.required_config_keys == {'n_channels', 'n_classes'}
        assert id(network_config) == id(alexnet.config)
        assert mock_validate_config.call_count == 1
        mock_validate_config.assert_called_with(
            network_config, AlexNet.required_config_keys
        )
        assert mock_set_layers.call_count == 1

    def test_set_layers(self, network_config):
        """Test _set_layers method

        This test checks that parameterized layers (convs & linear layers) are
        not set as attributes *before* `_set_layers` is called, but are set as
        attributes *after* `_set_layers` is called.

        :param network_config: network_config object fixture
        :type network_config: dict
        """

        alexnet = MagicMock()
        alexnet.config = network_config
        alexnet._set_layers = AlexNet._set_layers

        layer_names = [
            'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
            'linear1', 'linear2', 'linear3'
        ]

        # layers should just be MagicMock objects before calling `_set_layers`
        for layer_name in layer_names:
            assert isinstance(getattr(alexnet, layer_name), MagicMock)
        alexnet._set_layers(self=alexnet)
        for layer_name in layer_names:
            if 'conv' in layer_name:
                assert isinstance(
                    getattr(alexnet, layer_name), torch.nn.Conv2d
                )
            else:
                assert isinstance(
                    getattr(alexnet, layer_name), torch.nn.Linear
                )

    def test_forward(self, network_config, layers_dict):
        """Test forward method

        This checks that the right number of each of the different kinds of
        layers are called during a forward pass.

        :param network_config: network_config object fixture
        :type network_config: dict
        :param layers_dict: layers_dict object fixture
        :type layers_dict: dict
        """

        alexnet = MagicMock()
        alexnet.config = network_config
        alexnet.forward = AlexNet.forward

        inputs = torch.randn((1, 227, 227, 3))
        layer_names = [
            'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
            'linear1', 'linear2', 'linear3'
        ]
        for layer_name in layer_names:
            mock = MagicMock()
            # the layers just need to return something that is a tensor and can
            # be passed into ReLU, MaxPool2d, and Dropout layers, so the
            # `inputs` will work
            mock.return_value = inputs
            setattr(alexnet, layer_name, mock)

        wrapped_layer_contexts = []
        with ExitStack() as stack:
            for _, layer_dict in layers_dict.items():
                wrapped_layer_context = (
                    stack.enter_context(layer_dict['wrapped_layer'])
                )
                wrapped_layer_contexts.append(
                    (wrapped_layer_context, layer_dict['expected_count'])
                )
            alexnet.forward(self=alexnet, inputs=inputs)

            it = wrapped_layer_contexts
            for wrapped_layer_context, expected_call_count in it:
                assert (
                    wrapped_layer_context.call_count ==
                    expected_call_count
                )
