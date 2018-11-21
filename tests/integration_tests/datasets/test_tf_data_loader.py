"""Integration tests for datasets.tf_data_loader"""

import numpy as np
import tensorflow as tf

from datasets.imagenet_dataset import ImageNetDataSet
from datasets.ops import resize_images
from datasets.tf_data_loader import TFDataLoader
from utils.test_utils import df_images


class TestTFDataLoader(object):
    """Tests for TFDataLoader"""

    BATCH_SIZE = 2

    def _get_batches(self, tf_data_loader):
        """Return two batches of data from the provided `tf_data_loader`

        :param tf_data_loader: dataset to return batches from
        :type tf_data_loader: datasets.tf_data_loader.TFDataLoader
        :return: batches of data
        :rtype: list[numpy.ndarray]
        """

        # call `tf.reset_default_graph` to simulate the dataset being
        # instantiated twice during two different runs of training
        tf.reset_default_graph()

        with tf.device('/cpu:0'):
            tf.set_random_seed(1027)
            dataset = tf_data_loader.get_infinite_iter(
                batch_size=self.BATCH_SIZE
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

    def test_get_infinite_iter(self, df_images):
        """Test get_infinite_iter method

        This tests several things:
        - When `tf.set_random_seed` is used in the right place, the exact same
          batches are returned from the dataset returned by a TFDataLoader when
          it is used as a one shot iterator
        - The images in a single batch are centered
        - The images and labels are of the expected shapes

        :param df_images: df_images object fixture
        :type df_images: pandas.DataFrame
        """

        target_shape = (227, 227)
        map_ops = [
            (resize_images,
             {'size': target_shape, 'sample_keys': ['image']}),
            (tf.one_hot,
             {'sample_keys': ['label'], 'depth': 1000}),
            (tf.image.per_image_standardization,
             {'sample_keys': ['image']}),
        ]

        imagenet_dataset = ImageNetDataSet(df_images)
        tf_data_loader = TFDataLoader(imagenet_dataset, map_ops)

        batches1 = self._get_batches(tf_data_loader)
        batches2 = self._get_batches(tf_data_loader)

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
