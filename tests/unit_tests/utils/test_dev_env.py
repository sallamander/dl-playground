"""Unit tests for utils.dev_env"""

from utils import dev_env


def test_get():
    """Test `get` function"""

    value = dev_env.get('imagenet', 'dirpath_data')
    assert value == '/data/imagenet'

    value = dev_env.get('deep_lesion', 'dirpath_data')
    assert value == '/data/deep_lesion'
