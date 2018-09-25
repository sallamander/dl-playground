"""Tests for trainers.imagenet_trainer.py"""

import os
import tempfile

import imageio
import numpy as np
import pandas as pd
import pytest

from datasets.imagenet_dataset import ImageNetDataSet
from networks.alexnet import AlexNet
from trainers.imagenet_trainer import ImageNetTrainer


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

    @pytest.fixture(scope='class')
    def df_images(self):
        """df_images object fixture

        This will act as a df_images to pass to an ImageNetDataSet. It will
        contain three rows with two columns, `fpath_image` and `label`. Each
        `fpath_image` will point to a `numpy.ndarray` saved as a JPEG, and the
        `label` will be equal to the index of that row (i.e. 0, 1, and 2).

        :return: dataframe holding the filepath to the input image and the
        target label for the image
        :rtype: pandas.DataFrame
        """

        tempdir = tempfile.TemporaryDirectory()

        rows = []
        for idx in range(self.BATCH_SIZE):
            height, width = np.random.randint(128, 600, 2)
            num_channels = 3

            input_image = np.random.random((height, width, num_channels))
            fpath_image = os.path.join(tempdir.name, '{}.jpg'.format(idx))
            imageio.imwrite(fpath_image, input_image)

            rows.append({
                'fpath_image': fpath_image, 'label': idx
            })

        df_images = pd.DataFrame(rows)
        yield df_images

    @pytest.fixture(scope='class')
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

        :param trainer_config: trainer_config object fixture
        :type trainer_config: dict
        """

        imagenet_trainer = ImageNetTrainer(trainer_config)

        assert imagenet_trainer.optimizer == 'adam'
        assert imagenet_trainer.loss == 'categorical_crossentropy'
        assert imagenet_trainer.batch_size == self.BATCH_SIZE
        assert imagenet_trainer.num_epochs == 2

    def test_validate_config(self, trainer_config):
        """Test _validate_config method

        :param trainer_config: trainer_config object fixture
        :type trainer_config: dict
        """

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
