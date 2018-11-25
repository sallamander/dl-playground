"""Integration tests for datasets.pytorch_dataset_transformer"""

from torch import Tensor
from torchvision.transforms.functional import to_tensor

from datasets.imagenet_dataset import ImageNetDataSet
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

        transformations = [(to_tensor, {'sample_keys': ['image']})]
        imagenet_datsaet_transformed = PyTorchDataSetTransformer(
            numpy_dataset=imagenet_dataset, transformations=transformations
        )

        for idx in range(2):
            sample = imagenet_datsaet_transformed[idx]
            assert set(sample) == {'image', 'label'}
            assert sample['image'].shape == (3, 227, 227)
            assert isinstance(sample['image'], Tensor)
            assert sample['label'] == idx
