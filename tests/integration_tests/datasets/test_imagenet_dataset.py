"""Integration tests for datasets.imagenet"""

import numpy as np
import tensorflow as tf

from datasets.imagenet_dataset import ImageNetDataSet
from utils.test_utils import df_images


class TestImageNetDataSet(object):
    """Tests for ImageNetDataSet"""

    def _get_batches(self, imagenet_dataset):
        """Return two batches of data from the provided `imagenet_dataset`

        :param imagenet_dataset: dataset to return batches from
        :type imagenet_dataset: datasets.imagenet.ImageNetDataSet
        :return: batches of data
        :rtype: list[numpy.ndarray]
        """

        # call `tf.reset_default_graph` to simulate the dataset being
        # instantiated twice during two different runs of training
        tf.reset_default_graph()

        with tf.device('/cpu:0'):
            tf.set_random_seed(1027)
            dataset = imagenet_dataset.get_infinite_iter()
            iterator = dataset.make_one_shot_iterator()
            next_element_op = iterator.get_next()

        with tf.Session() as sess:
            # pull 2 batches to make sure it loops back over the dataset
            batches = []
            for _ in range(2):
                batch = sess.run(next_element_op)
                batches.append(batch)

        return batches

    def test_get_infinite_iter(self, df_images):
        """Test get_infinite_iter method

        This tests two things:
        - When `tf.set_random_seed` is used in the right place, the exact same
          batches are returned from ImageNetDataSet when it is used as a one
          shot iterator
        - The images in a single batch are centered

        :param df_images: df_images object fixture
        :type df_images: pandas.DataFrame
        """

        dataset_config = {'height': 227, 'width': 227, 'batch_size': 3}
        imagenet_dataset = ImageNetDataSet(df_images, dataset_config)

        batches1 = self._get_batches(imagenet_dataset)
        batches2 = self._get_batches(imagenet_dataset)

        for batch1, batch2 in zip(batches1, batches2):
            assert np.allclose(batch1[0], batch2[0])
            assert np.allclose(batch1[1], batch2[1])

        for batch in batches1:
            images = batch[0]
            for image in images:
                assert np.allclose(image.mean(), 0, atol=1e-4)
                assert np.allclose(image.std(), 1, atol=1e-4)
