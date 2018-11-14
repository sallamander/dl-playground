"""Unit tests for utils.generic_utils"""

import random

import numpy as np
import pandas as pd
import pytest

from utils.generic_utils import import_object, validate_config


class TestImportObject():
    """Test `import_object` function"""

    def test_import_object__dev_env(self, monkeypatch):
        """Test `import_object` with utils.dev_env"""

        def mock_dev_env_get(group, key):
            """Mock `utils.dev_env.get` function"""
            return '/data/imagenet'
        monkeypatch.setattr('utils.dev_env.get', mock_dev_env_get)

        get = import_object(object_importpath='utils.dev_env.get')
        assert get('imagenet', 'dirpath_data') == '/data/imagenet'
        assert get('mpii', 'dirpath_data') == '/data/imagenet'

    def test_import_object__numpy(self):
        """Test `import_object` with a numpy.array"""

        NumpyArray = import_object(object_importpath='numpy.array')
        array1 = NumpyArray([0, 1, 2, 3])
        array2 = np.array([0, 1, 2, 3])

        assert isinstance(array1, np.ndarray)
        assert np.array_equal(array1, array2)

    def test_import_object__pandas(self):
        """Test `import_object` with a pandas.DataFrame"""

        rows = [
            {'a': 0, 'b': 1, 'c': 2},
            {'a': 1, 'b': 2, 'c': 3},
            {'a': 2, 'b': 3, 'c': 4}
        ]

        PandasDataFrame = import_object(object_importpath='pandas.DataFrame')
        dataframe1 = PandasDataFrame(rows)
        dataframe2 = pd.DataFrame(rows)

        assert isinstance(dataframe1, pd.DataFrame)
        assert dataframe1.equals(dataframe2)


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
