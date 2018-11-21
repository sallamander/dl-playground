"""Tensorflow operations for tf.Dataset objects"""

import tensorflow as tf


def apply_transformation(transformation_fn, sample, sample_keys,
                         transformation_fn_kwargs=None):
    """Apply a transformation to certain elements of the provided sample

    :param transformation_fn: function to apply to each of the values located
     at the given `sample_keys` of `sample`
    :type transformation_fn: function
    :param sample: holds the elements to apply the `transformation_fn` to
    :type sample: dict
    :param sample_keys: holds the keys corresponding to the elements of
     `sample` to apply the `transformation_fn` to
    :type sample_keys: list[str]
    :param transformation_fn_kwargs: holds keyword arguments to pass to the
     `transformation_fn`
    :type transformation_fn_kwargs: dict
    :return: `sample` with `transformation_fn` applied to the specified
     elements
    """

    transformation_fn_kwargs = transformation_fn_kwargs or {}
    for key in sample_keys:
        sample[key] = transformation_fn(
            sample[key], **transformation_fn_kwargs
        )

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
