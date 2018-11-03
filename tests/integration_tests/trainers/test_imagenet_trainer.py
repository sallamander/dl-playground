"""Integration tests for trainers.imagenet_trainer"""

from datasets.imagenet_dataset import ImageNetDataSet
from networks.alexnet import AlexNet
from trainers.imagenet_trainer import ImageNetTrainer
from utils.test_utils import df_images


class TestImageNetTrainer(object):
    """Tests for ImageNetTrainer"""

    def test_train(self, df_images):
        """Test train method"""

        height = 227
        width = 227
        batch_size = 2

        network_config = {
            'height': height, 'width': width, 'n_channels': 3,
            'n_classes': 1000
        }
        dataset_config = {
            'height': height, 'width': width, 'batch_size': batch_size
        }
        trainer_config = {
            'optimizer': 'adam', 'loss': 'categorical_crossentropy',
            'batch_size': batch_size, 'num_epochs': 2
        }

        alexnet = AlexNet(network_config)
        imagenet_dataset = ImageNetDataSet(df_images, dataset_config)
        imagenet_trainer = ImageNetTrainer(trainer_config)

        imagenet_trainer.train(
            train_dataset=imagenet_dataset, val_dataset=imagenet_dataset,
            network=alexnet
        )
