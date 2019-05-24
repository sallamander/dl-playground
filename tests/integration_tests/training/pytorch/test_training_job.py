"""Integration tests for training.pytorch.training_job"""

import os
import tempfile
import pytest

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
import yaml

from constants import DIRPATH_DLP
from datasets.ops import per_image_standardization
from networks.pytorch.object_classification.alexnet import AlexNet
from training.pytorch.imagenet_trainer import ImageNetTrainer
from training.pytorch.training_job import PyTorchTrainingJob
from utils.test_utils import df_images


class TestPyTorchTrainingJob(object):
    """Tests for PyTorchTrainingJob"""

    @pytest.fixture(scope='class')
    def fpath_df_images(self, df_images):
        """Return df_images saved to a temporary file

        :return: filepath to saved df_images
        :rtype: str
        """

        df_images = df_images.copy()

        fpath_df_images = tempfile.mkstemp()[1]
        df_images.to_csv(fpath_df_images, index=False)
        yield fpath_df_images

    @pytest.fixture(scope='class')
    def config(self, fpath_df_images):
        """Return a `config` that specifies the training job

        :return: config file that specifies a PyTorch training job
        :rtype: dict
        """

        fpath_config = os.path.join(
            DIRPATH_DLP, 'training', 'training_configs',
            'alexnet_imagenet_pytorch.yml'
        )
        with open(fpath_config, 'r') as f:
            job_config = yaml.load(f, Loader=yaml.FullLoader)
        job_config['dataset']['fpath_df_train'] = fpath_df_images
        job_config['dataset']['train_loading_params']['batch_size'] = 2
        job_config['dataset']['fpath_df_validation'] = fpath_df_images
        job_config['dataset']['validation_loading_params']['batch_size'] = 1

        return job_config

    def test_instantiate_dataset(self, config):
        """Test _instantiate_dataset class

        :param config: config object fixture
        :type config: dict
        """

        job = PyTorchTrainingJob()
        job.config = config
        dataset, n_batches = job._instantiate_dataset(set_name='train')
        assert n_batches == 1
        assert isinstance(dataset, DataLoader)

        dataset, n_batches = job._instantiate_dataset(set_name='validation')
        assert n_batches == 3
        assert isinstance(dataset, DataLoader)

    def test_instantiate_network(self, config):
        """Test _instantiate_network class

        :param config: config object fixture
        :type config: dict
        """

        job = PyTorchTrainingJob()
        job.config = config
        network = job._instantiate_network()
        assert isinstance(network, AlexNet)
        assert network.config['n_channels'] == 3
        assert network.config['n_classes'] == 1000

    def test_instantiate_trainer(self, config):
        """Test _instantiate_trainer class

        :param config: config object fixture
        :type config: dict
        """

        job = PyTorchTrainingJob()
        job.config = config
        job.gpu_id = config['gpu_id']
        trainer = job._instantiate_trainer()
        assert isinstance(trainer, ImageNetTrainer)
        assert trainer.optimizer == 'Adam'
        assert trainer.loss == 'CrossEntropyLoss'
        assert trainer.batch_size == 32
        assert trainer.n_epochs == 10
        assert trainer.device == torch.device('cuda:0')

    def test_parse_transformations(self, config):
        """Test _parse_transformations method

        :param config: config object fixture
        :type config: dict
        """

        job = PyTorchTrainingJob()
        job.config = config
        for set_name in ['train', 'validation']:
            transformations_key = '{}_transformations'.format(set_name)
            transformations = config['dataset'][transformations_key]
            transformations = job._parse_transformations(transformations)

            assert len(transformations) == 3

            assert transformations[0][0] == per_image_standardization
            assert transformations[1][0] == to_tensor
            assert transformations[2][0] == torch.tensor

            assert transformations[0][1] == {'sample_keys': ['image']}
            assert transformations[1][1] == {'sample_keys': ['image']}
            assert (
                transformations[2][1] ==
                {'sample_keys': ['label'], 'dtype': torch.int64}
            )
