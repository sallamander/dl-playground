"""Unit tests for utils.generic_utils"""

from itertools import cycle as itertools_cycle
import random

import numpy as np
import pandas as pd
import pytest

from utils.generic_utils import cycle, import_object, validate_config


class TestCycle(object):
    """Test `cycle` function"""

    def test_cycle__bad_iterable(self):
        """Test `cycle` when the iterable doesn't implement `__iter__`"""

        def iterable():
            for element in range(5):
                yield element

        with pytest.raises(AttributeError):
            cycle_iter = cycle(iterable)
            for _ in range(5):
                next(cycle_iter)

    def test_cycle__deterministic(self):
        """Test `cycle` when the iterable yields elements deterministically"""

        class DeterministicIter(object):

            def __iter__(self):
                for element in range(5):
                    yield element

        cycle_iter = cycle(DeterministicIter())
        for idx_element in range(10):
            expected_element = idx_element % 5
            element = next(cycle_iter)
            assert element == expected_element

    def test_cycle__nondeterministic(self):
        """Test `cycle` when the iterable yields elements non-deterministically

        This compares the output of the `cycle` function with that of the
        `itertools.cycle` function. The former should return different sets of
        elements when cycling over the iterable twice, whereas the latter
        should return the same sets of elements, since it caches the results of
        the returned elements during the first pass.
        """

        class NonDeterministicIter(object):

            def __init__(self, seeds):
                self.cycle = 0
                self.seeds = seeds

            def __iter__(self):

                np.random.seed(self.seeds[self.cycle])
                for element in np.random.randint(0, 1000, size=5):
                    yield element
                self.cycle += 1

        itertools_cycle_iter = itertools_cycle(
            NonDeterministicIter(seeds=[1226, 42])
        )
        non_itertools_cycle_iter = cycle(
            NonDeterministicIter(seeds=[1226, 42])
        )

        itertools_batch1 = [next(itertools_cycle_iter) for _ in range(5)]
        itertools_batch2 = [next(itertools_cycle_iter) for _ in range(5)]

        non_itertools_batch1 = [
            next(non_itertools_cycle_iter) for _ in range(5)
        ]
        non_itertools_batch2 = [
            next(non_itertools_cycle_iter) for _ in range(5)
        ]

        assert np.array_equal(itertools_batch1, itertools_batch2)
        assert np.array_equal(itertools_batch1, non_itertools_batch1)
        assert not np.array_equal(non_itertools_batch1, non_itertools_batch2)


class TestImportObject(object):
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


class TestValidateConfig(object):
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
