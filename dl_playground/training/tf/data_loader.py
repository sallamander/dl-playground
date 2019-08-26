"""Loader to create a tensorflow.data.Dataset objects for training"""

import tensorflow as tf


def format_batch(batch, input_keys, target_keys):
    """Format the batch from a single dictionary into a tuple of dictionaries

    :param batch: batch of inputs and targets
	:type batch: dict[tensorflow.Tensor]
    :param input_keys: names of the keys in `batch` that are inputs to a model
	:type input_keys: list[str]
    :param target_keys: names of the keys in `batch` that are targets for a
     model
	:type target_keys: list[str]
    :return: 2-element tuple holding the inputs and targets
	:rtype: tuple(dict)
    """

    inputs = {input_key: batch[input_key] for input_key in input_keys}
    targets = {target_key: batch[target_key] for target_key in target_keys}

    return inputs, targets


class TFDataLoader(object):
    """Loader for batches of a tf.data.Dataset"""

    def __init__(self, augmented_dataset):
        """Init

        :param augmented_dataset: dataset that provides samples for training
        :type augmented_dataset: datasets.augmented_dataset.AugmentedDataset
        """

        self.augmented_dataset = augmented_dataset

    def get_infinite_iter(self, batch_size, shuffle=False,
                          prefetch_buffer_size=1, n_workers=0):
        """Return a tf.data.Dataset that iterates over the data indefinitely

        :param batch_size: size of the batches to return
        :type batch_size: int
        :param shuffle: if True, re-shuffle the data at the end of every epoch
        :type shuffle: bool
        :param prefetch_buffer_size: number of batches to prefetch
        :type prefetch_buffer_size: int
        :param n_workers: number of subprocesses to use for data loading
        :type n_workers: int
        :return: dataset that iterates over the data indefinitely
        :rtype: tensorflow.data.Dataset
        """

        generator = self.augmented_dataset.as_generator(
            shuffle=shuffle, n_workers=n_workers
        )
        sample_shapes = {
            name: tf.TensorShape(shape)
            for name, shape in self.augmented_dataset.sample_shapes.items()
        }
        dataset = tf.data.Dataset.from_generator(
            lambda: generator, self.augmented_dataset.sample_types,
            sample_shapes
        )
        dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            lambda batch: format_batch(
                batch, input_keys=self.augmented_dataset.input_keys,
                target_keys=self.augmented_dataset.target_keys
            )
        )
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset
