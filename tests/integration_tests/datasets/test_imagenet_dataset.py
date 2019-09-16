"""Integration tests for training.pytorch.dataset_transformer"""

import numpy as np
from torch import Tensor
from torchvision.transforms.functional import to_tensor
from trainet.datasets.augmented_dataset import AugmentedDataset
from trainet.datasets.ops import per_image_standardization
from trainet.utils.test_utils import df_images

from datasets.imagenet_dataset import ImageNetDataset


class TestImagenetDataset():
    """Tests for ImagenetDataset"""

    def test_getitem(self, df_images):
        """Test __getitem__ method

        :param df_images: df_images object fixture
        :type df_images: pandas.DataFrame
        """

        dataset_config = {'height': 227, 'width': 227}
        imagenet_dataset = ImageNetDataset(df_images, dataset_config)

        transformations = [
            (per_image_standardization, {}, {'image': 'image'}),
            (to_tensor, {}, {'image': 'pic'})
        ]
        augmented_dataset = AugmentedDataset(
            numpy_dataset=imagenet_dataset, transformations=transformations
        )

        for idx in range(2):
            sample = augmented_dataset[idx]
            image, label = sample['image'], sample['label']

            assert image.shape == (3, 227, 227)
            assert np.allclose(image.mean(), 0, atol=1e-6)
            assert np.allclose(image.std(), 1, atol=1e-6)
            assert isinstance(image, Tensor)

            assert label == idx
