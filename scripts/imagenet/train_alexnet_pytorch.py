#! /usr/bin/env python
"""Train AlexNet on ImageNet using pytorch"""

import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from training.pytorch.dataset_transformer import PyTorchDataSetTransformer
from utils import dev_env
from utils.generic_utils import import_object


DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
FPATH_DF_TRAIN_SET = os.path.join(
    DIRPATH_DATA, 'from_access_links',
    'metadata_lists', 'df_classification_train_set.csv'
)
FPATH_DF_VAL_SET = os.path.join(
    DIRPATH_DATA, 'from_access_links',
    'metadata_lists', 'df_classification_val_set.csv'
)


def get_data_loaders(dataset_spec):
    """Return train and validation data loaders

    :param dataset_spec: specifies how to build the train and validation
     datasets
    :type dataset_spec: dict
    :return: loaders of the training and validation data
    :rtype: tuple(torch.utils.data.DataLoader)
    """

    df_train = pd.read_csv(FPATH_DF_TRAIN_SET)
    df_val = pd.read_csv(FPATH_DF_VAL_SET)

    dataset_importpath = dataset_spec['importpath']
    DataSet = import_object(dataset_importpath)

    dataset_config = dataset_spec['init_params']['config']
    train_dataset = DataSet(df_train, dataset_config)
    val_dataset = DataSet(df_val, dataset_config)

    transformations = dataset_spec['transformations']
    processed_transformations = []
    for transformation in transformations:
        assert len(transformation) == 1
        transformation_fn_importpath = list(transformation.keys())[0]
        transformation_config = list(transformation.values())[0]

        transformation_fn = import_object(transformation_fn_importpath)
        processed_transformation_config = {}
        for param, arguments in transformation_config.items():
            value = arguments['value']
            if arguments.get('import'):
                value = import_object(value)
            processed_transformation_config[param] = value
        processed_transformations.append(
            (transformation_fn, processed_transformation_config)
        )

    train_dataset = PyTorchDataSetTransformer(
        numpy_dataset=train_dataset, transformations=processed_transformations
    )
    val_dataset = PyTorchDataSetTransformer(
        numpy_dataset=val_dataset, transformations=processed_transformations
    )

    train_loader = DataLoader(
        dataset=train_dataset, **dataset_spec['train_loader']
    )
    val_loader = DataLoader(
        dataset=val_dataset, **dataset_spec['val_loader']
    )

    return train_loader, val_loader


def get_network(network_spec):
    """Return an alexnet model to use during training

    :param network_spec: specifies how to build the AlexNet model
    :type network_spec: dict
    :return: alexnet model
    :rtype: networks.alexnet_pytorch.AlexNet
    """

    network_importpath = network_spec['importpath']
    Network = import_object(network_importpath)

    network_config = network_spec['init_params']['config']
    return Network(network_config)


def get_trainer(trainer_spec):
    """Return a trainer to train AlexNet on ImageNet

    :param trainer_spec: specifies how to train the AlexNet on ImageNet
    :type trainer_spec: dict
    :return: trainer to train alexnet on imagenet
    :rtype: trainers.imagenet_trainer_pytorch.ImageNetTrainer
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainer_importpath = trainer_spec['importpath']
    Trainer = import_object(trainer_importpath)

    trainer_config = trainer_spec['init_params']['config']
    trainer_config['device'] = device
    return Trainer(trainer_config)


def parse_args():
    """Parse command line arguments

    :return: namespace holding command line arguments
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fpath_config', type=str, required=True,
        help='Filepath to a training config.'
    )

    args = parser.parse_args()
    return args


def main():
    """Train AlexNet on ImageNet"""

    args = parse_args()
    with open(args.fpath_config) as f:
        training_config = yaml.load(f)

    train_loader, val_loader = get_data_loaders(training_config['dataset'])
    alexnet = get_network(training_config['network'])
    trainer = get_trainer(training_config['trainer'])

    trainer.train(
        network=alexnet, train_dataset=train_loader,
        n_steps_per_epoch=len(train_loader),
        validation_dataset=val_loader,
        n_validation_steps=len(val_loader)
    )


if __name__ == '__main__':
    main()
