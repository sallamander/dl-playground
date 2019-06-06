"""Integration tests for training.tf.imagenet_trainer"""

import tempfile

import tensorflow as tf
from tensorflow.python.keras.utils import to_categorical

from datasets.augmented_dataset import AugmentedDataset
from datasets.imagenet_dataset import ImageNetDataset
from datasets.ops import per_image_standardization
from networks.tf.object_classification.alexnet import AlexNet
from training.tf.data_loader import TFDataLoader
from training.tf.imagenet_trainer import ImageNetTrainer
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
        trainer_config = {
            'optimizer': 'adam', 'loss': 'categorical_crossentropy',
            'batch_size': batch_size, 'n_epochs': 2
        }
        dataset_config = {'height': height, 'width': width}
        transformations = [
            (to_categorical,
             {'sample_keys': ['label'], 'num_classes': 1000}),
            (per_image_standardization,
             {'sample_keys': ['image']}),
        ]

        alexnet = AlexNet(network_config)
        tempdir = tempfile.mkdtemp()
        imagenet_trainer = ImageNetTrainer(
            trainer_config, dirpath_save=tempdir
        )

        imagenet_dataset = ImageNetDataset(df_images, dataset_config)
        augmented_dataset = AugmentedDataset(imagenet_dataset, transformations)
        tf_data_loader = TFDataLoader(augmented_dataset)
        dataset = tf_data_loader.get_infinite_iter(
            batch_size=batch_size, shuffle=True, n_workers=4
        )

        imagenet_trainer.train(
            train_dataset=dataset, n_steps_per_epoch=len(imagenet_dataset),
            validation_dataset=dataset,
            n_validation_steps=len(imagenet_dataset), network=alexnet
        )
