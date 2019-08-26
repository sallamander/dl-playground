"""Integration tests for training.tf.data_loader"""

import pytest

import numpy as np
import torch
import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical

from datasets.augmented_dataset import AugmentedDataset
from datasets.imagenet_dataset import ImageNetDataset
from datasets.ops import per_image_standardization
from training.tf.data_loader import TFDataLoader
from utils.test_utils import df_images


class TestTFDataLoader(object):
    """Tests for TFDataLoader"""

    BATCH_SIZE = 2

    @pytest.fixture(scope='class')
    def tf_data_loader(self, df_images):
        """TFDataLoader object fixture

        :param df_images: df_images object fixture
        :type df_images: pandas.DataFrame
        :return: loader of the images in `df_images`
        :rtype: training.tf.data_loader.TFDataLoader
        """

        transformations = [
            (to_categorical,
             {'sample_keys': ['label'], 'num_classes': 1000}),
            (per_image_standardization,
             {'sample_keys': ['image']}),
        ]

        dataset_config = {'height': 227, 'width': 227}
        imagenet_dataset = ImageNetDataset(df_images, dataset_config)
        augmented_dataset = AugmentedDataset(imagenet_dataset, transformations)
        tf_data_loader = TFDataLoader(augmented_dataset)

        return tf_data_loader

    def _get_batches(self, tf_data_loader, shuffle=False):
        """Return two batches of data from the provided `tf_data_loader`

        :param shuffle: passed to tf_data_loader.get_infinite_iter
        :type shuffle: bool
        :param tf_data_loader: dataset to return batches from
        :type tf_data_loader: training.tf.data_loader.TFDataLoader
        :return: batches of data
        :rtype: list[numpy.ndarray]
        """

        # call `tf.reset_default_graph` to simulate the dataset being
        # instantiated twice during two different runs of training
        tf.reset_default_graph()

        with tf.device('/cpu:0'):
            tf.set_random_seed(1027)
            dataset = tf_data_loader.get_infinite_iter(
                batch_size=self.BATCH_SIZE, shuffle=shuffle
            )
            iterator = dataset.make_one_shot_iterator()
            next_element_op = iterator.get_next()

        with tf.Session() as sess:
            # pull 2 batches to make sure it loops back over the dataset
            batches = []
            for _ in range(2):
                batch = sess.run(next_element_op)
                batches.append(batch)

        return batches

    def test_get_infinite_iter__no_shuffling(self, tf_data_loader):
        """Test get_infinite_iter method when shuffling is equal to False

        This tests several things:
        - When shuffle=False, the exact same batches are returned from the
          dataset returned by a TFDataLoader when it is used as a one shot
          iterator
        - The images in a single batch are centered
        - The images and labels are of the expected shapes

        :param tf_data_loader: dataset to return batches from
        :type tf_data_loader: datasets.tf_data_loader.TFDataLoader
        """

        # the exact same batches should be returned even when the seed is set
        # differently
        torch.manual_seed(1)
        batches1 = self._get_batches(tf_data_loader, shuffle=False)
        torch.manual_seed(2)
        batches2 = self._get_batches(tf_data_loader, shuffle=False)

        for batch1, batch2 in zip(batches1, batches2):
            assert np.allclose(batch1[0]['image'], batch2[0]['image'])
            assert np.allclose(batch1[1]['label'], batch2[1]['label'])
            assert batch1[0]['image'].shape == (2, 227, 227, 3)
            assert batch1[1]['label'].shape == (2, 1000)

        for batch in batches1:
            images = batch[0]['image']
            for image in images:
                assert np.allclose(image.mean(), 0, atol=1e-4)
                assert np.allclose(image.std(), 1, atol=1e-4)
                assert image.shape == (227, 227, 3)

    def test_get_infinite_iter__shuffling(self, tf_data_loader):
        """Test get_infinite_iter method when shuffling is equal to True

        This tests several things:
        - When shuffle=True, different batches are returned from the dataset
          returned by a TFDataLoader when it is used as a one shot iterator
        - The images in a single batch are centered
        - The images and labels are of the expected shapes

        :param tf_data_loader: dataset to return batches from
        :type tf_data_loader: datasets.tf_data_loader.TFDataLoader
        """

        # since the dataset is only 3 elements, set the seed to test that
        # different results are returned when shuffle=True
        torch.manual_seed(1)
        batches1 = self._get_batches(tf_data_loader, shuffle=True)
        torch.manual_seed(2)
        batches2 = self._get_batches(tf_data_loader, shuffle=True)

        for batch1, batch2 in zip(batches1, batches2):
            assert not np.allclose(batch1[0]['image'], batch2[0]['image'])
            assert not np.allclose(batch1[1]['label'], batch2[1]['label'])
            assert batch1[0]['image'].shape == (2, 227, 227, 3)
            assert batch1[1]['label'].shape == (2, 1000)

        for batch in batches1:
            images = batch[0]['image']
            for image in images:
                assert np.allclose(image.mean(), 0, atol=1e-4)
                assert np.allclose(image.std(), 1, atol=1e-4)
                assert image.shape == (227, 227, 3)
