"""Unit tests for datasets.imagenet_dataset_np"""

from unittest.mock import MagicMock
import pytest

import numpy as np
import imageio

from datasets.imagenet_dataset import ImageNetDataSet
from utils.test_utils import df_images


class TestImageNetDataSet(object):
    """Tests for ImageNetDataSet"""

    def test_init(self, df_images):
        """Test __init__ method

        This tests two things:
        - All attributes are set correctly in the __init__
        - A KeyError is raised if the fpath_image or label column is missing in
          the `df_images` passed to the __init__ of the ImageNetDataSet
        - The `sample_types` property is returned correctly

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        """

        # === test all attributes are set correctly === #
        dataset = ImageNetDataSet(df_images)
        assert dataset.df_images.equals(df_images)

        # === test `df_images` === #
        for col in ['fpath_image', 'label']:
            df_images_underspecified = df_images.drop(col, axis=1)

            with pytest.raises(KeyError):
                ImageNetDataSet(df_images_underspecified)

        # === test `sample_types` === #
        expected_sample_types = {'image': 'float32', 'label': 'uint8'}
        assert dataset.sample_types == expected_sample_types

    def test_len(self, df_images):
        """Test __len__ method

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        """

        imagenet_dataset = MagicMock()
        imagenet_dataset.df_images = df_images
        imagenet_dataset.__len__ = ImageNetDataSet.__len__

        assert len(imagenet_dataset) == 3

    def test_get_item(self, df_images):
        """Test __getitem__ method

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        """

        sample_types = {'image': 'float16', 'label': 'int16'}
        df_images['label'] = df_images['label'].astype(sample_types['label'])

        imagenet_dataset = MagicMock()
        imagenet_dataset.df_images = df_images
        imagenet_dataset.sample_types = sample_types
        imagenet_dataset.__getitem__ = ImageNetDataSet.__getitem__

        for idx in range(3):
            sample = imagenet_dataset[idx]

            assert sample['image'].dtype == sample_types['image']
            assert sample['label'].dtype == sample_types['label']

            assert sample['label'] == df_images.loc[idx, 'label']
            assert np.array_equal(
                sample['image'],
                imageio.imread(df_images.loc[idx, 'fpath_image'])
            )

        with pytest.raises(KeyError):
            imagenet_dataset[4]

    def test_as_generator(self):
        """Test `as_generator` method"""

        def mock_get_item(self, idx):
            """Mock __getitem__ magic method"""
            return idx

        def mock_len(self):
            """Mock __len__ magic method"""
            return 9

        imagenet_dataset = MagicMock()
        imagenet_dataset.__getitem__ = mock_get_item
        imagenet_dataset.__len__ = mock_len
        imagenet_dataset.as_generator = ImageNetDataSet.as_generator

        gen = imagenet_dataset.as_generator(self=imagenet_dataset)
        dataset = [element for element in gen]
        assert np.array_equal(range(9), dataset)
