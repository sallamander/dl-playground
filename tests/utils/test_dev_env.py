"""Tests for utils.dev_env.py"""

from utils import dev_env


def test_get():
    """Test `get` function"""

    value = dev_env.get('imagenet', 'dirpath_data')
    assert value == '/data/imagenet'