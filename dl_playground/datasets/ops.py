"""Dataset ops"""

import numpy as np


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


def per_image_standardization(image):
    """Return the provided `image` with zero mean and unit variance

    This mimics the `tensorflow.image.per_image_standardization`
    implementation, located at
    https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization.

    :param image: image data to standardize, of shape (height, width,
     n_channels)
    :type image: numpy.ndarray
    :return: `image` standardized to have zero mean and unit variance
    :rtype: numpy.ndarray
    """

    if image.ndim != 3:
        msg = '`image` must have 3 dimensions, but has shape {}'
        raise ValueError(msg.format(image.shape))

    image_mean = image.mean()
    min_std = 1 / np.sqrt(image.size)
    image_std = max(image.std(), min_std)

    image = (image - image_mean) / image_std
    return image
