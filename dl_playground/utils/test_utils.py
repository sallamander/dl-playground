"""Testing utilities for dl-playground integration and unit tests."""

import os
import tempfile
import pytest

import imageio
import numpy as np
import pandas as pd


@pytest.fixture(scope='class')
def df_images():
    """Return a `df_images` pointing to temporarily saved images

    `df_images` will contain three rows with two columns, `fpath_image` and
    `label`. Each `fpath_image` will point to a `numpy.ndarray` saved as a
    JPEG, and the `label` will be equal to the index of that row (i.e. 0, 1,
    and 2).

    :return: dataframe holding the filepath to the input image and the target
     label for the image
    :rtype: pandas.DataFrame
    """

    tempdir = tempfile.TemporaryDirectory()

    rows = []
    for idx in range(3):
        height, width = np.random.randint(128, 600, 2)
        num_channels = np.random.choice((1, 4), 1).tolist()[0]

        input_image = np.random.random((height, width, num_channels))
        fpath_image = os.path.join(tempdir.name, '{}'.format(idx))
        if num_channels <= 3:
            fpath_image += '.jpg'
        else:
            fpath_image += '.tif'
        imageio.imwrite(fpath_image, input_image)

        rows.append({
            'fpath_image': fpath_image, 'label': idx
        })

    df_images = pd.DataFrame(rows)
    yield df_images
