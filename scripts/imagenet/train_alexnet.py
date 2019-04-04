#! /usr/bin/env python
"""Train AlexNet on ImageNet"""

import argparse

import yaml

from utils.generic_utils import import_object


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

    TrainingJob = import_object(training_config['job_importpath'])
    training_job = TrainingJob(training_config)
    training_job.run()


if __name__ == '__main__':
    main()
