#! /usr/bin/env python
"""Train AlexNet on ImageNet"""

import os

import pandas as pd

from datasets.imagenet_dataset import ImageNetDataSet
from networks.alexnet import AlexNet
from trainers.imagenet_trainer import ImageNetTrainer
from utils import dev_env


DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
FPATH_DF_TRAIN_SET = os.path.join(
    DIRPATH_DATA, 'from_access_links', 'metadata_lists', 'df_train_set.csv'
)
FPATH_DF_VAL_SET = os.path.join(
    DIRPATH_DATA, 'from_access_links', 'metadata_lists', 'df_val_set.csv'
)

IMAGE_HEIGHT = 227
IMAGE_WIDTH = 227
BATCH_SIZE = 128


def get_datasets():
    """Return train and val datasets

    :return: datasets to pass to an ImageNetTrainer object
    :rtype: tuple(datasets.imagenet_dataset.ImageNetDataSet)
    """

    df_train = pd.read_csv(FPATH_DF_TRAIN_SET)
    df_val = pd.read_csv(FPATH_DF_VAL_SET)

    dataset_config = {
        'height': IMAGE_HEIGHT, 'width': IMAGE_WIDTH,
        'batch_size': BATCH_SIZE
    }
    train_dataset = ImageNetDataSet(df_train, dataset_config)
    val_dataset = ImageNetDataSet(df_val, dataset_config)

    return train_dataset, val_dataset


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
    :rtype: trainers.imagenet_trainer.ImageNetTrainer
    """

    trainer_config = {
        'optimizer': 'adam', 'loss': 'categorical_crossentropy',
        'batch_size': BATCH_SIZE, 'num_epochs': 10
    }
    return ImageNetTrainer(trainer_config)


def main():
    """Train AlexNet on ImageNet"""

    train_dataset, val_dataset = get_datasets()
    network = get_network()
    trainer = get_trainer()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    trainer.train(train_dataset, network, val_dataset)


if __name__ == '__main__':
    main()
