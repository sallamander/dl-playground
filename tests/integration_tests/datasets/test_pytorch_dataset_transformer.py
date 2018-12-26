"""Integration tests for datasets.pytorch_dataset_transformer"""

import numpy as np
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from datasets.imagenet_dataset import ImageNetDataSet
from datasets.ops import per_image_standardization
from datasets.pytorch_dataset_transformer import PyTorchDataSetTransformer
from utils.test_utils import df_images


class TestPyTorchDataSetTransformer(object):
    """Tests for PyTorchDataSetTransformer"""

    def test_getitem(self, df_images):
        """Test __getitem__ method

        :param df_images: df_images object fixture
        :type df_images: pandas.DataFrame
        """

        dataset_config = {'height': 227, 'width': 227}
        imagenet_dataset = ImageNetDataSet(df_images, dataset_config)

        transformations = [
            (per_image_standardization, {'sample_keys': ['image']}),
            (to_tensor, {'sample_keys': ['image']})
        ]
        imagenet_datsaet_transformed = PyTorchDataSetTransformer(
            numpy_dataset=imagenet_dataset, transformations=transformations
        )

        for idx in range(2):
            image, label = imagenet_datsaet_transformed[idx]

            assert image.shape == (3, 227, 227)
            assert np.allclose(image.mean(), 0, atol=1e-6)
            assert np.allclose(image.std(), 1, atol=1e-6)
            assert isinstance(image, Tensor)

            assert label == idx
