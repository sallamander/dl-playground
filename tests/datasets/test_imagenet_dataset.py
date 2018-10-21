"""Tests for datasets.imagenet"""

import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

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
            'height': 227, 'width': 227, 'batch_size': 256,
        }

    def _check_batches(self, batch_size, dataset_config, batches):
        """Assert the size of the inputs and outputs of `batches`

        :param dataset_config: dataset_config object fixture, with 'batch_size'
         key equal to `batch_size`
        :type dataset_config: dict
        :param batch_size: size of the batch
        :type batch_size: int
        :param batches: batches of input and output pairs
        :type batches: list[numpy.ndarray]
        """

        expected_image_shape = (
            dataset_config['height'], dataset_config['width'], 3
        )
        expected_target_shape = (1000, )

        for batch in batches:
            assert batch[0].shape == (batch_size, ) + expected_image_shape
            assert batch[1].shape == (batch_size, ) + expected_target_shape

        assert not np.allclose(batches[0][0], batches[1][0])
        assert not np.allclose(batches[0][1], batches[1][1])

    def _get_batches(self, imagenet_dataset):
        """Return two batches of data from the provided `imagenet_dataset`

        :param imagenet_dataset: dataset to return batches from
        :type imagenet_dataset: datasets.imagenet.ImageNetDataSet
        :return: batches of data
        :rtype: list[numpy.ndarray]
        """

        # call `tf.reset_default_graph` is necessary to ensure that the
        # `tf.set_random_seed` call produces the same batching results
        tf.reset_default_graph()

        with tf.device('/cpu:0'):
            # with such a small batch size (df_images is small), its important
            # to set the random seed to actually ensure that the batches that
            # come out are different
            tf.set_random_seed(911)
            dataset = imagenet_dataset.get_infinite_iter()
            iterator = dataset.make_one_shot_iterator()
            next_element_op = iterator.get_next()

        with tf.Session() as sess:
            # pull 2 batches to ensure that it loops back over the dataset
            batches = []
            for _ in range(2):
                batch = sess.run(next_element_op)
                batches.append(batch)

        return batches

    def test_init(self, df_images, dataset_config):
        """Test __init__ method

        This tests three things:
        - All attributes are set correctly in the __init__
        - A KeyError is raised if the fpath_image or label column is missing in
          the `df_images` passed to the __init__ of the ImageNetDataSet
        - A KeyError is raised if 'height', 'width', or 'batch_size' are not
          present in the dataset_config

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        """

        # === test all attributes are set correctly === #
        dataset = ImageNetDataSet(df_images, dataset_config)

        assert df_images.equals(dataset.df_images)
        assert dataset.height == 227
        assert dataset.width == 227
        assert dataset.batch_size == 256
        assert dataset.num_parallel_calls == 4

        # === test `df_images` === #
        for col in ['fpath_image', 'label']:
            df_images_underspecified = df_images.drop(col, axis=1)

            with pytest.raises(KeyError):
                ImageNetDataSet(df_images_underspecified, dataset_config)

        # === test `dataset_config` === #
        for dataset_key in dataset_config:
            dataset_config_copy = dataset_config.copy()
            del dataset_config_copy[dataset_key]

            with pytest.raises(KeyError):
                ImageNetDataSet(df_images, dataset_config_copy)

    def test_len(self, df_images, dataset_config):
        """Test __len__ magic method

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        """

        imagenet_dataset = ImageNetDataSet(df_images, dataset_config)
        assert len(imagenet_dataset) == 3

        df_images2 = pd.concat([df_images] * 2)
        imagenet_dataset = ImageNetDataSet(df_images2, dataset_config)
        assert len(imagenet_dataset) == 6

    def test_get_infinite_iter(self, df_images, dataset_config):
        """Test get_infinite_iter method

        :param df_images : df_images object fixture
        :type: pandas.DataFrame
        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        """

        # make a copy and adjust the batch_size for this test
        dataset_config = dataset_config.copy()
        batch_size = len(df_images)
        dataset_config['batch_size'] = batch_size
        imagenet_dataset = ImageNetDataSet(df_images, dataset_config)

        # call self._get_batches() twice to verify that using
        # tf.set_random_seed ensures that the exact same batches are returned
        batches1 = self._get_batches(imagenet_dataset)
        self._check_batches(batch_size, dataset_config, batches1)
        batches2 = self._get_batches(imagenet_dataset)

        # assert that the batches are the exact same
        for batch_idx in range(len(batches1)):
            assert np.allclose(batches1[batch_idx][0], batches2[batch_idx][0])
            assert np.allclose(batches1[batch_idx][1], batches2[batch_idx][1])

        # check that each individual input is centered
        images = batches1[0][0]
        for image in images:
            assert np.allclose(image.mean(), 0, atol=1e-4)
            assert np.allclose(image.std(), 1, atol=1e-4)
