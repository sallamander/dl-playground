"""Tensorflow operations for tf.Dataset objects"""

import tensorflow as tf


def apply_op(map_op, sample, sample_keys, map_op_kwargs=None):
    """Apply an operation to certain elements of the provided sample

    :param map_op: function to apply to each of the values located at the given
     `sample_keys` of `sample`
    :type map_op: function
    :param sample: holds the elements to apply the `map_op` to
    :type sample: dict
    :param sample_keys: holds the keys corresponding to the elements of
     `sample` to apply the `map_op` to
    :type sample_keys: list[str]
    :param map_op_kwargs: holds keyword arguments to pass to the `map_op`
    :type map_op_kwargs: dict
    :return: `sample` with `map_op` applied to the specified elements
    """

    map_op_kwargs = {} if map_op_kwargs is None else map_op_kwargs
    for key in sample_keys:
        sample[key] = map_op(sample[key], **map_op_kwargs)

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
