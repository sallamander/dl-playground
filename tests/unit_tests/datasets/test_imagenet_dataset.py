"""Unit tests for datasets.imagenet_dataset_np"""

from unittest.mock import MagicMock
import pytest

import imageio
import numpy as np
import torch

from datasets.imagenet_dataset import ImageNetDataSet
from utils.test_utils import df_images


class TestImageNetDataSet(object):
    """Tests for ImageNetDataSet"""

    @pytest.fixture(scope='class')
    def dataset_config(self):
        """dataset_config object fixture

        :return: dataset_config to be used for ImageNet training
        :rtype: dict
        """

        return {'height': 227, 'width': 227}

    def test_init(self, df_images, dataset_config, monkeypatch):
        """Test __init__ method

        This tests several things:
        - All attributes are set correctly in the __init__
        - A KeyError is raised if the fpath_image or label column is missing in
          the `df_images` passed to the __init__ of the ImageNetDataSet
        - The `sample_types` property is returned correctly

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        :param dataset_config: dataset_config object fixture
        :type: dict
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_validate_config = MagicMock()
        monkeypatch.setattr(
            'datasets.imagenet_dataset.validate_config', mock_validate_config
        )

        # === test all attributes are set correctly === #
        dataset = ImageNetDataSet(df_images, dataset_config)
        assert dataset.df_obs.equals(df_images)
        assert dataset.config == dataset_config
        assert dataset.required_config_keys == {'height', 'width'}
        mock_validate_config.assert_called_once_with(
            dataset.config, dataset.required_config_keys
        )

        # === test `df_images` === #
        for col in ['fpath_image', 'label']:
            df_images_underspecified = df_images.drop(col, axis=1)

            with pytest.raises(KeyError):
                ImageNetDataSet(df_images_underspecified, dataset_config)

        # === test `sample_types` === #
        expected_sample_types = {'image': 'float32', 'label': 'uint8'}
        assert dataset.sample_types == expected_sample_types

    def test_len(self, df_images):
        """Test __len__ method

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        """

        imagenet_dataset = MagicMock()
        imagenet_dataset.df_obs = df_images
        imagenet_dataset.__len__ = ImageNetDataSet.__len__

        assert len(imagenet_dataset) == 3

    def test_getitem(self, df_images, dataset_config):
        """Test __getitem__ method

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        :param dataset_config: dataset_config object fixture
        :type: dict
        """

        sample_types = {'image': 'float16', 'label': 'int16'}
        df_images['label'] = df_images['label'].astype(sample_types['label'])

        imagenet_dataset = MagicMock()
        imagenet_dataset.df_obs = df_images
        imagenet_dataset.config = dataset_config
        imagenet_dataset.sample_types = sample_types
        imagenet_dataset.__getitem__ = ImageNetDataSet.__getitem__

        for idx in range(3):
            sample = imagenet_dataset[idx]

            assert sample['image'].dtype == sample_types['image']
            assert sample['label'].dtype == sample_types['label']

            assert sample['label'] == df_images.loc[idx, 'label']
            assert sample['image'].shape == (227, 227, 3)

            fpath_image = df_images.loc[idx, 'fpath_image']
            image = imageio.imread(fpath_image)
            if image.ndim == 2:
                sample_image = sample['image']
                assert np.array_equal(
                    sample_image[..., 0], sample_image[..., 1]
                )
                assert np.array_equal(
                    sample_image[..., 1], sample_image[..., 2]
                )
            elif image.shape[-1] == 4:
                sample_image = sample['image']
                assert sample_image.shape[-1] == 3

        with pytest.raises(KeyError):
            imagenet_dataset[4]

    def test_as_generator(self):
        """Test `as_generator` method

        This tests two cases:
        - `shuffle=False`
        - `shuffle=True`

        It only tests when `n_workers=0`, deferring the testing of
        `n_workers=1` to integration tests.
        """

        def mock_get_item(self, idx):
            """Mock __getitem__ magic method"""
            return {idx: idx}

        def mock_len(self):
            """Mock __len__ magic method"""
            return 9

        imagenet_dataset = MagicMock()
        imagenet_dataset.__getitem__ = mock_get_item
        imagenet_dataset.__len__ = mock_len
        imagenet_dataset.as_generator = ImageNetDataSet.as_generator

        # set the seed to one where we know that the shuffle won't by random
        # chance produce the unshuffled version
        torch.manual_seed(1)
        for shuffle in [True, False]:
            gen = imagenet_dataset.as_generator(
                self=imagenet_dataset, n_workers=0, shuffle=shuffle
            )
            dataset = [next(gen) for _ in range(9)]
            indices, values = [], []
            for sample in dataset:
                for idx, value in sample.items():
                    indices.append(idx)
                    values.append(value.tolist())

            if shuffle:
                assert indices != list(range(9))
                assert values != list(range(9))
            else:
                assert indices == list(range(9))
                assert values == list(range(9))
