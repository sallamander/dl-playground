#! /usr/bin/env python
"""Train AlexNet on ImageNet"""

import argparse
import os

import yaml

from utils.generic_utils import import_object


def parse_args():
    """Parse command line arguments

    :return: namespace holding command line arguments
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser()

    run_or_resume_group = parser.add_mutually_exclusive_group(required=True)
    run_or_resume_group.add_argument(
        '--fpath_config', type=str, required=False,
        help='Filepath to a training config.'
    )
    run_or_resume_group.add_argument(
        '--dirpath_job', type=str, required=False,
        help='Directory path to a previously started training job.'
    )

    args = parser.parse_args()
    return args


def main():
    """Train AlexNet on ImageNet"""

    args = parse_args()

    if args.fpath_config:
        fpath_config = args.fpath_config
        resume = False
    else:
        fpath_config = os.path.join(
            args.dirpath_job, 'config.yml'
        )
        resume = True

    with open(fpath_config) as f:
        training_config = yaml.load(f, Loader=yaml.FullLoader)

    TrainingJob = import_object(training_config['job_importpath'])
    training_job = TrainingJob()

    if resume:
        training_job.resume(args.dirpath_job)
    else:
        training_job.run(training_config)


if __name__ == '__main__':
    main()
