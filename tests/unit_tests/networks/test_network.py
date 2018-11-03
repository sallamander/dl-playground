"""Unit tests for networks.network"""

from unittest.mock import MagicMock

from networks.network import Network


class TestNetwork(object):
    """Tests for Network"""

    def test_init(self, monkeypatch):
        """Test __init__

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        monkeypatch.setattr(Network, 'required_config_keys', {'test1'})

        mock_validate_config = MagicMock()
        monkeypatch.setattr(
            'networks.network.validate_config', mock_validate_config
        )

        network_configs = [
            {'height': 227, 'width': 227,
             'n_classes': 3, 'n_channels': 3},
            {'test1': 'key1', 'test2': 'key2'}
        ]

        it = enumerate(network_configs, 1)
        for expected_call_count, network_config in it:
            network = Network(network_config)
            assert network_config == network.network_config
            assert mock_validate_config.call_count == expected_call_count
