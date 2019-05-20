"""Loader to create a tensorflow.data.Dataset objects for training"""

import tensorflow as tf

from datasets.ops import apply_transformation, format_batch


class TFDataLoader(object):
    """Loader for batches of a tf.data.Dataset"""

    def __init__(self, numpy_dataset, transformations=None):
        """Init

        :param numpy_dataset: dataset that provides samples for training
        :type numpy_dataset: torch.utils.data.Dataset object
        :param transformations: holds 2 element tuples with the first element
         being a function to apply to the dataset samples and the second
         element being a dictionary of keyword arguments to pass to those
         functions
        :type transformations: list[tuple(function, dict)]
        """

        self.numpy_dataset = numpy_dataset
        self.transformations = transformations or []

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

        generator = self.numpy_dataset.as_generator(
            shuffle=shuffle, n_workers=n_workers
        )
        output_shapes = {
            name: tf.TensorShape(shape)
            for name, shape in self.numpy_dataset.output_shapes.items()
        }
        dataset = tf.data.Dataset.from_generator(
            lambda: generator, self.numpy_dataset.sample_types,
            output_shapes
        )
        dataset = dataset.repeat()

        it = self.transformations
        for transformation_fn, transformation_fn_kwargs in it:
            transformation_fn_kwargs = transformation_fn_kwargs.copy()
            sample_keys = transformation_fn_kwargs.pop('sample_keys')

            dataset = dataset.map(
                lambda sample: apply_transformation(
                    transformation_fn, sample,
                    sample_keys, transformation_fn_kwargs
                )
            )

        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            lambda batch: format_batch(
                batch, input_keys=self.numpy_dataset.input_keys,
                target_keys=self.numpy_dataset.target_keys
            )
        )
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset
