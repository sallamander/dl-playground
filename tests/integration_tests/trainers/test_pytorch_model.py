"""Integration tests for trainers.pytorch_model"""

from unittest.mock import patch

from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from datasets.imagenet_dataset import ImageNetDataSet
from datasets.ops import per_image_standardization
from datasets.pytorch_dataset_transformer import PyTorchDataSetTransformer
from networks.alexnet_pytorch import AlexNet
from trainers.pytorch_model import Model
from utils.generic_utils import cycle
from utils.test_utils import df_images


class TestModel(object):
    """Tests for Model"""

    def test_fit_generator(self, df_images):
        """Test fit_generator_method"""

        dataset_config = {'height': 227, 'width': 227}
        train_dataset = ImageNetDataSet(df_images, dataset_config)
        transformations = [
            (per_image_standardization, {'sample_keys': ['image']}),
            (to_tensor, {'sample_keys': ['image']}),
        ]
        train_dataset = PyTorchDataSetTransformer(
            numpy_dataset=train_dataset, transformations=transformations
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=2,
            shuffle=True, num_workers=0
        )

        alexnet = AlexNet(network_config={'n_channels': 3, 'n_classes': 1000})

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
                generator=cycle(train_loader),
                n_steps_per_epoch=2, n_epochs=2
            )

            assert patched_loss.call_count == 4
