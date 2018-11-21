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


def center_image(image, label):
    """Apply tf.image.per_image_standardization to `image`

    Note that `label` remains unchanged.

    :param image: image to center
    :type image: tensorflow.Tensor
    :param label: label to pass through
    :type label: tensorflow.Tensor
    :return: tensorflow.Tensor objects holding the centered image and
     unmodified label
    :rtype: tuple(tensorflow.Tensor)
    """

    image = tf.image.per_image_standardization(image)

    return image, label


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


def load_image(fpath_image, label):
    """Parse /decode and return the provided image

    This op is intended to be used on a tf.Dataset object that is built around
    image filepaths as the inputs and class labels as the targets, where the
    image filepaths point to 2 dimensional RGB images (i.e. of shape (height,
    width, 3)).

    :param fpath_image: filepath to the input image to parse
    :type fpath_image: str
    :param label: class label associated with an image; this will simply be
     passed through this function
    :type label: tensorflow.Tensor
    :return: tensorflow.Tensor objects holding the parsed / decoded image
     along with the class label
    :rtype: tuple(tensorflow.Tensor)
    """

    image_string = tf.read_file(fpath_image)
    image = tf.image.decode_image(image_string, channels=3)

    # set_shape because `tf.image.decode_image` does not set the shape, and if
    # it isn't set then tf.image_resize_images won't work downstream
    image.set_shape([None, None, None])
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


def reshape_image_and_label(image, label, target_image_shape,
                            num_label_classes):
    """Reshape image to target_image_shape and one-hot encode label

    :param image: image to reshape
    :type image: tensorflow.Tensor
    :param label: label to reshape
    :type label: tensorflow.Tensor
    :param target_image_shape: (height, width) to resize `image` to
    :type target_image_shape: tuple or list
    :param num_label_classes: `dense` argument to pass to `tensorflow.one_hot`
    :type num_label_classes: int
    :return: tensorflow.Tensor objects holding the reshaped image and label
    :rtype: tuple(tensorflow.Tensor)
    """

    image = tf.image.resize_images(image, target_image_shape)
    label = tf.one_hot(label, num_label_classes)

    return image, label


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
