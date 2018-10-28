"""Unit tests for datasets.imagenet"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from datasets.imagenet_dataset import ImageNetDataSet
from utils.test_utils import df_images


class TestImageNetDataSet(object):
    """Tests for ImageNetDataSet"""

    @pytest.fixture(scope='class')
    def dataset_config(self):
        """dataset_config object fixture

        :return: dataset_config to be used for Imagenet training
        :rtype: dict
        """

        return {
            'height': 227, 'width': 227, 'batch_size': 3,
        }

    def _check_batches(self, dataset_config, batches):
        """Assert the size of the inputs and outputs of `batches`

        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        :param batches: batches of input and output pairs
        :type batches: list[numpy.ndarray]
        """

        expected_image_shape = (
            dataset_config['height'], dataset_config['width'], 3
        )
        expected_target_shape = (1000, )

        batch_size = dataset_config['batch_size']
        for batch in batches:
            assert batch[0].shape == (batch_size, ) + expected_image_shape
            assert batch[1].shape == (batch_size, ) + expected_target_shape

    def _get_batches(self, imagenet_dataset):
        """Return two batches of data from the provided `imagenet_dataset`

        :param imagenet_dataset: dataset to return batches from
        :type imagenet_dataset: datasets.imagenet.ImageNetDataSet
        :return: batches of data
        :rtype: list[numpy.ndarray]
        """

        with tf.device('/cpu:0'):
            dataset = imagenet_dataset.get_infinite_iter(self=imagenet_dataset)
            iterator = dataset.make_initializable_iterator()
            next_element_op = iterator.get_next()

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            # pull 2 batches to ensure that it loops back over the dataset
            batches = []
            for _ in range(2):
                batch = sess.run(next_element_op)
                batches.append(batch)

        return batches

    def test_init(self, df_images, dataset_config, monkeypatch):
        """Test __init__ method

        This tests two things:
        - All attributes are set correctly in the __init__
        - A KeyError is raised if the fpath_image or label column is missing in
          the `df_images` passed to the __init__ of the ImageNetDataSet

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        assert ImageNetDataSet.required_config_keys == {
            'height', 'width', 'batch_size'
        }

        def mock_validate_config(config, required_keys):
            """Mock validate_config to pass"""
            pass
        monkeypatch.setattr(
            'datasets.imagenet_dataset.validate_config', mock_validate_config
        )

        # === test all attributes are set correctly === #
        dataset = ImageNetDataSet(df_images, dataset_config)

        assert df_images.equals(dataset.df_images)
        assert dataset.height == 227
        assert dataset.width == 227
        assert dataset.batch_size == 3
        assert dataset.num_parallel_calls == 4

        # === test `df_images` === #
        for col in ['fpath_image', 'label']:
            df_images_underspecified = df_images.drop(col, axis=1)

            with pytest.raises(KeyError):
                ImageNetDataSet(df_images_underspecified, dataset_config)

    def test_len(self, df_images):
        """Test __len__ magic method

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        """

        imagenet_dataset = MagicMock()
        imagenet_dataset.__len__ = ImageNetDataSet.__len__
        imagenet_dataset.df_images = df_images

        assert len(imagenet_dataset) == 3

        df_images2 = pd.concat([df_images] * 2)
        imagenet_dataset.df_images = df_images2
        assert len(imagenet_dataset) == 6

    def test_get_infinite_iter(self, df_images, dataset_config, monkeypatch):
        """Test get_infinite_iter method

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        def mock_load_image(fpath_image, label):
            """Mock validate_config function"""

            height, width = np.random.randint(128, 600, 2)
            num_channels = 3
            image = np.random.random((height, width, num_channels))

            return image, label

        def mock_reshape_image_and_label(image, label, target_image_shape,
                                         num_label_classes=1000):
            """Mock reshape_image_and_label function"""

            num_channels = (3, )
            target_image_shape = tuple(target_image_shape)

            image = np.random.random(target_image_shape + num_channels)
            label = np.random.randint(0, num_label_classes, num_label_classes)

            return image, label

        def mock_center_image(image, label):
            """Mock center_image to simply return the image and label"""
            return image, label

        monkeypatch.setattr(
            'datasets.imagenet_dataset.load_image', mock_load_image
        )
        monkeypatch.setattr(
            'datasets.imagenet_dataset.reshape_image_and_label',
            mock_reshape_image_and_label
        )
        monkeypatch.setattr(
            'datasets.imagenet_dataset.center_image', mock_center_image
        )

        imagenet_dataset = MagicMock()
        imagenet_dataset.get_infinite_iter = ImageNetDataSet.get_infinite_iter
        imagenet_dataset.batch_size = dataset_config['batch_size']
        imagenet_dataset.height = dataset_config['height']
        imagenet_dataset.width = dataset_config['width']
        imagenet_dataset.num_parallel_calls = 3
        imagenet_dataset.df_images = df_images

        batches = self._get_batches(imagenet_dataset)
        self._check_batches(dataset_config, batches)
