"""Tensorflow operations for tf.Dataset objects"""

import tensorflow as tf


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
    :rtype: tuple
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
    """

    image = tf.image.resize_images(image, target_image_shape)
    label = tf.one_hot(label, num_label_classes)

    return image, label
