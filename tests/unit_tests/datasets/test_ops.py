"""Unit tests for datasets.ops"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from datasets.ops import apply_transformation, format_batch, resize_images


@pytest.fixture(scope='module')
def image():
    """Return a tensorflow.Tensor holding an image object fixture

    :return: tensorflow.Tensor holding an image to use in tests
    :rtype: tensorflow.Tensor
    """

    height, width = np.random.randint(128, 600, 2)
    num_channels = 3
    image = np.random.random((height, width, num_channels))
    image = tf.constant(image)

    return image


@pytest.fixture(scope='module')
def label():
    """Return a tensorflow.Tensor holding an label object fixture

    :return: tensorflow.Tensor holding a label to use in tests
    :rtype: tensorflow.Tensor
    """

    label = tf.constant(1, dtype=tf.uint8)
    return label


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


def test_resize_images():
    """Test resize_images"""

    images = tf.placeholder(tf.float32, name='images')

    target_shape = (227, 227)
    resize_images_op = resize_images(images, target_shape)
    with tf.Session() as sess:
        resized_images = sess.run(
            resize_images_op, feed_dict={images: np.ones((128, 64, 3))}
        )
    assert resized_images.shape == (227, 227, 3)

    images = tf.placeholder(tf.float32, name='images')
    images.set_shape = MagicMock()
    with pytest.raises(ValueError):
        resize_images_op = resize_images(images, target_shape)


class TestApplyTransformation(object):
    """Tests for `apply_transformation` over different use cases"""

    def test_apply_transformation__image_centering(self, image, label):
        """Test `apply_transformation` with `tf.image.per_image_standardization`

        This only tests the centering of the 'image' key in the `sample` below,
        but 'label' is still included to simulate a more realistic scenario
        where the `sample` has both an 'image' and 'label' key.

        :param image: module wide image object fixture
        :type image: tensorflow.Tensor
        :param label: module wide label object fixture
        :type label: tensorflow.Tensor
        """

        sample = {'image': image, 'label': label}
        sample_keys = {'image'}
        transformation_fn = tf.image.per_image_standardization

        sample_centered_op = apply_transformation(
            transformation_fn, sample, sample_keys
        )

        with tf.Session() as sess:
            sample_centered = sess.run(sample_centered_op)

        assert sample_centered['image'].shape == sample['image'].shape
        assert np.allclose(sample_centered['image'].mean(), 0, atol=1e-4)
        assert np.allclose(sample_centered['image'].std(), 1, atol=1e-4)
        assert sample_centered['label'] == 1

    @pytest.mark.parametrize('num_classes', [1000, 10, 3, 2])
    def test_apply_transformation__one_hot(self, image, label, num_classes):
        """Test `apply_transformation` with `tf.one_hot`

        This only tests that `tf.one_hot` is applied correctly to the 'label'.
        The 'image' is still included to simulate a more realistic scenario
        where the `sample` has both an 'image' and a 'label' key.

        :param image: module wide image object fixture
        :type image: tensorflow.Tensor
        :param label: module wide label object fixture
        :type label: tensorflow.Tensor
        :param num_classes: number of classes to use as the `depth` argument to
         `tf.one_hot`
        :type number_classes: int
        """

        sample = {'image': image, 'label': label}
        sample_keys = {'label'}
        transformation_fn = tf.one_hot
        transformation_fn_kwargs = {'depth': num_classes}

        sample_one_hotted_op = apply_transformation(
            transformation_fn, sample, sample_keys, transformation_fn_kwargs
        )

        with tf.Session() as sess:
            sample_one_hotted = sess.run(sample_one_hotted_op)

        assert sample_one_hotted['image'].shape == sample['image'].shape
        assert sample_one_hotted['label'].shape == (num_classes, )
        assert sample_one_hotted['label'].argmax() == 1

    @pytest.mark.parametrize('target_image_shape', [
        (227, 227), (64, 64), (128, 196), (64, 32)
    ])
    def test_apply_transformation__resize_images(self, image, label,
                                                 target_image_shape):
        """Test `apply_transformation` with `tf.image_resize_images`

        `label` is not directly used in the testing, but is still included to
        simulate a more realistic scenario where the `sample` has a 'label'
        key.

        :param image: module wide image object fixture
        :type image: tensorflow.Tensor
        :param label: module wide label object fixture
        :type label: tensorflow.Tensor
        :param target_image_shape: (height, width) to reshape `image` to
        :type target_image_shape: tuple(int)
        """

        sample = {'image1': image, 'image2': image, 'label': label}
        sample_keys = {'image1', 'image2'}
        transformation_fn = tf.image.resize_images
        transformation_fn_kwargs = {'size': target_image_shape}

        sample_resized_op = apply_transformation(
            transformation_fn, sample, sample_keys, transformation_fn_kwargs
        )

        with tf.Session() as sess:
            sample_resized = sess.run(sample_resized_op)

        num_channels = (3, )
        expected_target_shape = target_image_shape + num_channels
        assert sample_resized['image1'].shape == expected_target_shape
        assert sample_resized['image2'].shape == expected_target_shape
        assert sample_resized['label'] == 1
