#! /usr/bin/env python
"""Train AlexNet on ImageNet using pytorch"""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from datasets.imagenet_dataset import ImageNetDataSet
from datasets.ops import per_image_standardization
from datasets.pytorch_dataset_transformer import PyTorchDataSetTransformer
from networks.alexnet_pytorch import AlexNet
from trainers.imagenet_trainer_pytorch import ImageNetTrainer
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
NUM_EPOCHS = 10


def get_data_loaders():
    """Return train and validation data loaders

    :return: loaders of the training and validation data
    :rtype: tuple(torch.utils.data.DataLoader)
    """

    df_train = pd.read_csv(FPATH_DF_TRAIN_SET)
    df_val = pd.read_csv(FPATH_DF_VAL_SET)

    dataset_config = {'height': IMAGE_HEIGHT, 'width': IMAGE_WIDTH}
    train_dataset = ImageNetDataSet(df_train, dataset_config)
    val_dataset = ImageNetDataSet(df_val, dataset_config)

    transformations = [
        (per_image_standardization, {'sample_keys': ['image']}),
        (to_tensor, {'sample_keys': ['image']}),
        (torch.tensor, {'sample_keys': ['label'], 'dtype': torch.long})
    ]
    train_dataset = PyTorchDataSetTransformer(
        numpy_dataset=train_dataset, transformations=transformations
    )
    val_dataset = PyTorchDataSetTransformer(
        numpy_dataset=val_dataset, transformations=transformations
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )

    return train_loader, val_loader


def get_network():
    """Return an alexnet model to use during training

    :return: alexnet model
    :rtype: networks.alexnet_pytorch.AlexNet
    """

    network_config = {'n_channels': 3, 'n_classes': 1000}
    return AlexNet(network_config)


def get_trainer():
    """Return a trainer to train AlexNet on ImageNet

    :return: trainer to train alexnet on imagenet
    :rtype: trainers.imagenet_trainer_pytorch.ImageNetTrainer
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer_config = {
        'optimizer': 'Adam', 'loss': 'CrossEntropyLoss',
        'batch_size': BATCH_SIZE, 'n_epochs': 10, 'device': device
    }
    return ImageNetTrainer(trainer_config)


def main():
    """Train AlexNet on ImageNet"""

    train_loader, val_loader = get_data_loaders()
    alexnet = get_network()
    trainer = get_trainer()

    trainer.train(
        network=alexnet, train_dataset=train_loader,
        n_steps_per_epoch=len(train_loader.dataset),
        validation_dataset=val_loader,
        n_validation_steps=len(val_loader.dataset)
    )


if __name__ == '__main__':
    main()
