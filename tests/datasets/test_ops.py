"""Tests for datasets.ops"""

import os
import tempfile
import pytest

import imageio
import numpy as np
import tensorflow as tf

from datasets.ops import center_image, load_image, reshape_image_and_label


@pytest.fixture(scope='module')
def image():
    """Return a tensorflow.Tensor holding an image object fixture

    :return: tensorflow.Tensor holding an image to use in tests
    :rtype: tensorflow.Tensor
    """

    height, width = np.random.randint(128, 600, 2)
    num_channels = 3
    image = np.random.random((height, width, num_channels))

    return image


@pytest.fixture(scope='module')
def label():
    """Return a tensorflow.Tensor holding an label object fixture

    :return: tensorflow.Tensor holding a label to use in tests
    :rtype: tensorflow.Tensor
    """

    label = tf.constant(1, dtype=tf.uint8)
    return label


def test_center_image(image, label):
    """Test center_image

    :param image: module wide image object fixture
    :type image: tensorflow.Tensor
    :param lable: module wide label object fixture
    :type label: tensorflow.Tensor
    """

    image_centered_op, label = center_image(image, label)

    with tf.Session() as sess:
        image_centered = sess.run(image_centered_op)

        assert np.allclose(image_centered.mean(), 0, atol=1e-4)
        assert np.allclose(image_centered.std(), 1, atol=1e-4)


def test_load_image():
    """Test load_image"""

    tempdir = tempfile.TemporaryDirectory()

    fpath_jpg = os.path.join(tempdir.name, 'image.jpg')
    fpath_png = os.path.join(tempdir.name, 'image.png')

    height, width = np.random.randint(128, 600, 2)
    num_channels = 3
    image = np.random.random((height, width, num_channels)).astype(np.uint8)
    imageio.imwrite(fpath_jpg, image)
    imageio.imwrite(fpath_png, image)

    for fpath_image in [fpath_jpg, fpath_png]:
        with tf.device('/cpu:0'):
            loaded_image_op, label_op = load_image(
                fpath_image, label=tf.constant(1)
            )

        with tf.Session() as sess:
            loaded_image = sess.run(loaded_image_op)
            label = sess.run(label_op)

            assert np.allclose(loaded_image, image)
            assert label == 1


def test_reshape_image_and_label(image, label):
    """Test reshape_image_and_label

    :param image: module wide image object fixture
    :type image: tensorflow.Tensor
    :param lable: module wide label object fixture
    :type label: tensorflow.Tensor
    """

    sess = tf.Session()
    test_cases = [
        {'target_image_shape': (64, 64), 'num_label_classes': 1},
        {'target_image_shape': [28, 32], 'num_label_classes': 3},
        {'target_image_shape': (128, 64), 'num_label_classes': 1000},
        {'target_image_shape': [256, 256], 'num_label_classes': 10},
    ]
    for test_case in test_cases:
        target_image_shape = test_case['target_image_shape']
        num_label_classes = test_case['num_label_classes']
        with tf.device('/cpu:0'):
            image_reshaped_op, label_reshaped_op = reshape_image_and_label(
                image, label, target_image_shape, num_label_classes
            )

        image_reshaped = sess.run(image_reshaped_op)
        label_reshaped = sess.run(label_reshaped_op)

        expected_image_shape = tuple(target_image_shape) + (3, )
        assert np.allclose(image_reshaped.shape, expected_image_shape)
        assert np.allclose(label_reshaped.shape, (num_label_classes,))

    sess.close()
