#! /usr/bin/env python
"""Train AlexNet on ImageNet using Tensorflow"""

import os

import pandas as pd
import tensorflow as tf

from datasets.imagenet_dataset import ImageNetDataSet
from networks.tf.object_classification.alexnet import AlexNet
from training.tf.data_loader import TFDataLoader
from training.tf.imagenet_trainer import ImageNetTrainer
from utils import dev_env


DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
FPATH_DF_TRAIN_SET = os.path.join(
    DIRPATH_DATA, 'from_access_links',
    'metadata_lists', 'df_classification_train_set.csv'
)
FPATH_DF_VAL_SET = os.path.join(
    DIRPATH_DATA, 'from_access_links',
    'metadata_lists', 'df_classification_val_set.csv'
)

IMAGE_HEIGHT = 227
IMAGE_WIDTH = 227
BATCH_SIZE = 128


def get_data_loaders():
    """Return train and validation data loaders

    :return: loaders of the training and validation data
    :rtype: tuple(datasets.tf_data_loader.TFDataLoader)
    """

    df_train = pd.read_csv(FPATH_DF_TRAIN_SET)
    df_val = pd.read_csv(FPATH_DF_VAL_SET)

    dataset_config = {'height': IMAGE_HEIGHT, 'width': IMAGE_WIDTH}
    train_dataset = ImageNetDataSet(df_train, dataset_config)
    validation_dataset = ImageNetDataSet(df_val, dataset_config)

    transformations = [
        (tf.one_hot,
         {'sample_keys': ['label'], 'depth': 1000}),
        (tf.image.per_image_standardization,
         {'sample_keys': ['image']})
    ]
    train_loader = TFDataLoader(train_dataset, transformations)
    validation_loader = (
        TFDataLoader(validation_dataset, transformations)
    )

    return train_loader, validation_loader


def get_network():
    """Return an alexnet model to use during training

    :return: alexnet model
    :rtype: networks.alexnet.AlexNet
    """

    network_config = {
        'height': IMAGE_HEIGHT, 'width': IMAGE_WIDTH,
        'n_channels': 3, 'n_classes': 1000
    }
    return AlexNet(network_config)


def get_trainer():
    """Return a trainer to train AlexNet on ImageNet

    :return: trainer to train alexnet on imagenet
    :rtype: trainers.imagenet_trainer_tf.ImageNetTrainer
    """

    trainer_config = {
        'optimizer': 'adam', 'loss': 'categorical_crossentropy',
        'batch_size': BATCH_SIZE, 'n_epochs': 10
    }
    return ImageNetTrainer(trainer_config)


def main():
    """Train AlexNet on ImageNet"""

    train_loader, validation_loader = get_data_loaders()
    alexnet = get_network()
    trainer = get_trainer()

    train_dataset = train_loader.get_infinite_iter(BATCH_SIZE, shuffle=True)
    validation_dataset = validation_loader.get_infinite_iter(BATCH_SIZE)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    trainer.train(
        network=alexnet,
        train_dataset=train_dataset,
        n_steps_per_epoch=len(train_loader.numpy_dataset) // BATCH_SIZE,
        validation_dataset=validation_dataset,
        n_validation_steps=len(validation_loader.numpy_dataset) // BATCH_SIZE
    )

if __name__ == '__main__':
    main()
