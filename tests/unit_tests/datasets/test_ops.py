"""Unit tests for datasets.ops"""

import numpy as np
import pytest
import tensorflow as tf

from datasets.ops import apply_transformation, per_image_standardization


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


class TestPerImageStandardization(object):
    """Test `per_image_standardization`"""

    def test_per_image_standardization(self):
        """Test `per_image_standardization` on a non-uniform image"""

        image = np.random.random((227, 227, 3))
        image_standardized = per_image_standardization(image)

        assert np.allclose(image_standardized.mean(), 0)
        assert np.allclose(image_standardized.std(), 1)

        with pytest.raises(ValueError):
            image = np.random.random((1, 2, 3, 4))
            image_standardized = per_image_standardization(image)

    def test_per_image_standardization__uniform(self):
        """Test `per_image_standardization` on a uniform image

        The main point of this test is to ensure that there is no division by
        zero because of the uniformity of the image. In this case, we expect
        that the standard deviation of the pixel values will be 0, but that the
        resulting mean will still also be 0.
        """

        image = np.ones((227, 227, 3))
        image_standardized = per_image_standardization(image)

        assert np.allclose(image_standardized.mean(), 0)
        assert np.allclose(image_standardized.std(), 0)
