"""Integration tests for training.pytorch.model"""

import tempfile
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from datasets.imagenet_dataset import ImageNetDataSet
from datasets.ops import per_image_standardization
from networks.pytorch.object_classification.alexnet import AlexNet
from training.pytorch.dataset_transformer import PyTorchDataSetTransformer
from training.pytorch.model import Model
from utils.generic_utils import cycle
from utils.test_utils import df_images


class TestModel(object):
    """Tests for Model"""

    def test_load_weights(self):
        """Test load_weights method

        This tests that a model saved using the `Model.save_weights` method can
        be loaded using the `Model.load_weights` method.
        """

        alexnet = AlexNet(config={'n_channels': 3, 'n_classes': 1000})
        model1 = Model(network=alexnet)
        model2 = Model(network=alexnet)

        fpath_weights = tempfile.mktemp()
        model1.save_weights(fpath_weights)
        model2.load_weights(fpath_weights)

    def test_fit_generator(self, df_images):
        """Test fit_generator method"""

        dataset_config = {'height': 227, 'width': 227}
        dataset = ImageNetDataSet(df_images, dataset_config)
        transformations = [
            (per_image_standardization, {'sample_keys': ['image']}),
            (to_tensor, {'sample_keys': ['image']}),
            (torch.tensor, {'sample_keys': ['label'], 'dtype': torch.long})
        ]
        dataset = PyTorchDataSetTransformer(
            numpy_dataset=dataset, transformations=transformations
        )
        data_loader = DataLoader(
            dataset=dataset, batch_size=2,
            shuffle=True, num_workers=4
        )

        alexnet = AlexNet(config={'n_channels': 3, 'n_classes': 1000})
        model = Model(network=alexnet)

        assert not model._compiled
        assert not model.optimizer
        assert not model.loss
        model.compile(optimizer='Adam', loss='CrossEntropyLoss')
        assert model._compiled
        assert model.optimizer
        assert model.loss

        with patch.object(model, 'loss', wraps=model.loss) as patched_loss:
            model.fit_generator(
                generator=cycle(data_loader),
                n_steps_per_epoch=2, n_epochs=2,
                validation_data=cycle(data_loader), n_validation_steps=3
            )

            assert patched_loss.call_count == 10
