"""Integration tests for trainers.imagenet_trainer"""

import tensorflow as tf

from datasets.imagenet_dataset import ImageNetDataSet
from datasets.ops import resize_images
from datasets.tf_data_loader import TFDataLoader
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
        trainer_config = {
            'optimizer': 'adam', 'loss': 'categorical_crossentropy',
            'batch_size': batch_size, 'num_epochs': 2
        }
        map_ops = [
            (resize_images,
             {'size': (height, width), 'sample_keys': ['image']}),
            (tf.one_hot,
             {'sample_keys': ['label'], 'depth': 1000}),
            (tf.image.per_image_standardization,
             {'sample_keys': ['image']}),
        ]

        alexnet = AlexNet(network_config)
        imagenet_trainer = ImageNetTrainer(trainer_config)

        imagenet_dataset = ImageNetDataSet(df_images)
        tf_data_loader = TFDataLoader(imagenet_dataset)
        tf_data_loader = TFDataLoader(imagenet_dataset, map_ops)
        dataset = tf_data_loader.get_infinite_iter(
            batch_size=batch_size
        )

        imagenet_trainer.train(
            train_dataset=dataset, steps_per_epoch=len(imagenet_dataset),
            validation_dataset=dataset, validation_steps=len(imagenet_dataset),
            network=alexnet
        )
