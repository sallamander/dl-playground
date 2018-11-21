"""Unit tests for datasets.tf_data_loader"""

from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf

from datasets.tf_data_loader import TFDataLoader
from utils.test_utils import df_images


class TestTFDataLoader(object):
    """Tests for TFDataLoader"""

    BATCH_SIZE = 2

    def _get_batches(self, tf_data_loader):
        """Return two batches of data from the provided `tf_data_loader`

        :param tf_data_loader: dataset to return batches from
        :type tf_data_loader: datasets.imagenet_dataset_tf.TFDataset
        :return: batches of data
        :rtype: list[numpy.ndarray]
        """

        with tf.device('/cpu:0'):
            dataset = tf_data_loader.get_infinite_iter(
                self=tf_data_loader, batch_size=self.BATCH_SIZE,
            )
            iterator = dataset.make_initializable_iterator()
            next_element_op = iterator.get_next()

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            # pull 2 batches to ensure that it loops back over the dataset
            batches = []
            for _ in range(3):
                batch = sess.run(next_element_op)
                batches.append(batch)

        return batches

    def test_init(self):
        """Test __init__ method

        This tests all attributes are set correctly in the __init__.
        """

        numpy_dataset = MagicMock()
        tf_data_loader = TFDataLoader(numpy_dataset=numpy_dataset)
        assert id(tf_data_loader.numpy_dataset) == id(numpy_dataset)
        assert not tf_data_loader.map_ops

        map_ops = MagicMock()
        tf_data_loader = TFDataLoader(
            numpy_dataset=numpy_dataset, map_ops=map_ops
        )
        assert id(tf_data_loader.map_ops) == id(map_ops)

    def test_get_infinite_iter(self):
        """Test get_infinite_iter method"""

        def mock_call():
            """Mock __call__ magic method"""

            for element in np.arange(4, dtype='float32'):
                yield {'element': element, 'label': 1}

        def return_max_val(element, ceiling):
            """Return the max of the element and ceiling"""
            return tf.maximum(element, ceiling)

        def add_to_label(label):
            """Add 1 to the provided label"""
            return label + 1

        map_ops = [
            (add_to_label, {'sample_keys': ['label']}),
            (return_max_val, {'sample_keys': ['element'], 'ceiling': 10})
        ]

        numpy_dataset = MagicMock()
        numpy_dataset.as_generator = mock_call
        numpy_dataset.sample_types = {'element': 'float32', 'label': 'int16'}
        numpy_dataset.input_keys = ['element']
        numpy_dataset.target_keys = ['label']

        tf_data_loader = MagicMock()
        tf_data_loader.numpy_dataset = numpy_dataset
        tf_data_loader.map_ops = map_ops
        tf_data_loader.get_infinite_iter = TFDataLoader.get_infinite_iter

        batches = self._get_batches(tf_data_loader)
        assert len(batches) == 3

        for batch in batches:
            assert np.array_equal(batch[0]['element'], [10., 10.])
            assert np.array_equal(batch[1]['label'], [2, 2])
