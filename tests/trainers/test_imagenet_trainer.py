"""Tests for trainers.imagenet_trainer.py"""

import pytest

from datasets.imagenet_dataset import ImageNetDataSet
from networks.alexnet import AlexNet
from trainers.imagenet_trainer import ImageNetTrainer
from utils.test_utils import df_images


class TestImageNetTrainer(object):
    """Tests for ImageNetTrainer"""

    BATCH_SIZE = 3
    HEIGHT = 227
    WIDTH = 227

    @pytest.fixture(scope='class')
    def alexnet(self):
        """alexnet object fixture

        :return: alexnet model to use during training
        :rtype: AlexNet
        """

        network_config = {
            'height': self.HEIGHT, 'width': self.WIDTH,
            'n_channels': 3, 'n_classes': 1000
        }
        return AlexNet(network_config)

    @pytest.fixture
    def imagenet_dataset(self, df_images):
        """imagenet_dataset object fixture

        :return: imagenet_dataset to be used in training
        :rtype: ImageNetDataSet
        """

        dataset_config = {
            'height': self.HEIGHT, 'width': self.WIDTH,
            'batch_size': self.BATCH_SIZE
        }
        imagenet_dataset = ImageNetDataSet(df_images, dataset_config)
        return imagenet_dataset

    @pytest.fixture(scope='class')
    def trainer_config(self):
        """trainer_config object fixture"""

        return {
            'optimizer': 'adam', 'loss': 'categorical_crossentropy',
            'batch_size': self.BATCH_SIZE, 'num_epochs': 2
        }

    def test_init(self, trainer_config):
        """Test __init__ method

        This tests two things:
        - All attributes are set correctly in the __init__
        - A KeyError is raised if 'optimizer', 'loss', 'batch_size', or
          'num_epochs' is not present in the `trainer_config`

        :param trainer_config: trainer_config object fixture
        :type trainer_config: dict
        """

        # === test all attributes are set correctly === #
        imagenet_trainer = ImageNetTrainer(trainer_config)

        assert imagenet_trainer.optimizer == 'adam'
        assert imagenet_trainer.loss == 'categorical_crossentropy'
        assert imagenet_trainer.batch_size == self.BATCH_SIZE
        assert imagenet_trainer.num_epochs == 2

        # === test `trainer_config` === #
        for trainer_key in trainer_config:
            trainer_config_copy = trainer_config.copy()
            del trainer_config_copy[trainer_key]

            with pytest.raises(KeyError):
                ImageNetTrainer(trainer_config_copy)

    def test_train(self, trainer_config, imagenet_dataset, alexnet):
        """Test train method

        :param trainer_config: trainer_config object fixture
        :type trainer_config: dict
        :param imagenet_dataset: imagenet_dataset object fixture
        :type imagenet_dataset: datasets.imagenet_dataset.ImageNetDataSet
        :param alexnet: alexnet object fixture
        :type alexnet: networks.alexnet.AlexNet
        """

        imagenet_trainer = ImageNetTrainer(trainer_config)
        imagenet_trainer.train(
            train_dataset=imagenet_dataset, network=alexnet
        )

        imagenet_trainer.train(
            train_dataset=imagenet_dataset, network=alexnet,
            val_dataset=imagenet_dataset
        )
