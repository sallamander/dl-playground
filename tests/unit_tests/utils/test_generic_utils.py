"""Unit tests for utils.generic_utils"""

import random

import pytest

from utils.generic_utils import validate_config


class TestValidateConfig():
    """Test `validate_config` function"""

    def test_validate_config__good(self):
        """Test `validate_config` when all required keys are present"""

        possible_keys = [(1, 2), 'required1', 85, 'required2', 4, (2, 3, 4)]

        for _ in range(3):
            required_keys = random.sample(possible_keys, 3)
            config = {
                required_key: 'sentinel_value'
                for required_key in required_keys
            }

            validate_config(config, required_keys)

    def test_validate_config__bad(self):
        """Test `validate_config` when required keys are missing"""

        config = {0: 1, 'test': 'testy_test', (0, 1): 4}
        required_keys = list(config.keys())

        for key in config:
            config_copy = config.copy()
            del config_copy[key]

            with pytest.raises(KeyError):
                validate_config(config_copy, required_keys)
