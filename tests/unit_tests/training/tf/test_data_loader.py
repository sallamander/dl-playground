"""Unit tests for training.tf.data_loader"""

from unittest.mock import MagicMock

import numpy as np
import tensorflow as tf

from training.tf.data_loader import format_batch, TFDataLoader
from utils.generic_utils import cycle
from utils.test_utils import df_images


def test_format_batch():
    """Test format_batch"""

    height, width = np.random.randint(128, 600, 2)
    batch_size = np.random.randint(2, 4)
    num_channels = 3
    images = np.random.random((batch_size, height, width, num_channels))
    labels = np.random.randint(0, 1000, batch_size)

    batch = {'images': images, 'labels': labels}
    formatted_batch = format_batch(
        batch, input_keys=['images'], target_keys=['labels']
    )
    assert len(formatted_batch) == 2
    assert list(formatted_batch[0]) == ['images']
    assert np.array_equal(formatted_batch[0]['images'], images)
    assert list(formatted_batch[1]) == ['labels']
    assert np.array_equal(formatted_batch[1]['labels'], labels)

    images2 = images + 5
    labels2 = labels + 2
    batch = {'images': images, 'images2': images2,
             'labels': labels, 'labels2': labels2}
    formatted_batch = format_batch(
        batch, input_keys=['images', 'labels'],
        target_keys=['images2', 'labels2']
    )
    assert len(formatted_batch) == 2

    assert len(formatted_batch[0]) == 2
    assert len(formatted_batch[1]) == 2
    assert set(formatted_batch[0]) == {'images', 'labels'}
    assert set(formatted_batch[1]) == {'images2', 'labels2'}

    assert np.array_equal(formatted_batch[0]['images'], images)
    assert np.array_equal(formatted_batch[0]['labels'], labels)
    assert np.array_equal(formatted_batch[1]['images2'], images2)
    assert np.array_equal(formatted_batch[1]['labels2'], labels2)


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
                self=tf_data_loader, batch_size=self.BATCH_SIZE, n_workers=0
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

        augmented_dataset = MagicMock()
        tf_data_loader = TFDataLoader(augmented_dataset=augmented_dataset)
        assert id(tf_data_loader.augmented_dataset) == id(augmented_dataset)

    def test_get_infinite_iter(self):
        """Test get_infinite_iter method"""

        def mock_call(shuffle=False, n_workers=1):
            """Mock __call__ magic method"""

            for element in cycle(np.arange(4, dtype='float32')):
                yield {'element': element, 'label': 1}

        augmented_dataset = MagicMock()
        augmented_dataset.as_generator = mock_call
        augmented_dataset.sample_types = {'element': 'float32', 'label': 'int16'}
        augmented_dataset.input_keys = ['element']
        augmented_dataset.target_keys = ['label']
        augmented_dataset.sample_shapes = {'element': (), 'label': ()}

        tf_data_loader = MagicMock()
        tf_data_loader.augmented_dataset = augmented_dataset
        tf_data_loader.get_infinite_iter = TFDataLoader.get_infinite_iter

        batches = self._get_batches(tf_data_loader)
        assert len(batches) == 3

        for idx, batch in enumerate(batches):
            idx = idx % 2
            assert np.array_equal(
                batch[0]['element'], [idx * 2, (idx * 2) + 1.]
            )
            assert np.array_equal(batch[1]['label'], [1, 1])
