"""Tensorflow operations for tf.Dataset objects"""

import tensorflow as tf


def apply_op(map_op_fn, sample, sample_keys, map_op_fn_kwargs=None):
    """Apply an operation to certain elements of the provided sample

    :param map_op_fn: function to apply to each of the values located at the
     given `sample_keys` of `sample`
    :type map_op_fn: function
    :param sample: holds the elements to apply the `map_op_fn` to
    :type sample: dict
    :param sample_keys: holds the keys corresponding to the elements of
     `sample` to apply the `map_op_fn` to
    :type sample_keys: list[str]
    :param map_op_fn_kwargs: holds keyword arguments to pass to the `map_op_fn`
    :type map_op_fn_kwargs: dict
    :return: `sample` with `map_op_fn` applied to the specified elements
    """

    map_op_fn_kwargs = {} if map_op_fn_kwargs is None else map_op_fn_kwargs
    for key in sample_keys:
        sample[key] = map_op_fn(sample[key], **map_op_fn_kwargs)

    return sample


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
    return (inputs, targets)


def resize_images(images, size):
    """Wrapper around tf.image.resize_images to resolve unknown shape errors

    :param images: 4-D tensor of shape (batch, height, width, n_channels) or a
     3-D tensor of shape (height, with, n_channels)
    :type image: tensorflow.Tensor
    :param size: the new (height, width) to resize `images` to
    :type size: iterable of ints
    :return: `images` resized
    :rtype: tensorflow.Tensor
    """

    images.set_shape((None, None, None))
    images = tf.image.resize_images(images, size=size)
    return images
