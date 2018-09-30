#! /usr/bin/env python
"""Plot example images from the training set"""

import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing
import os

import imageio
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from utils import dev_env

DIRPATH_IMAGENET = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_CLASSIFICATION_PLOTS = os.path.join(
    DIRPATH_IMAGENET, 'classification_plots', 'training_set', 'ground_truth'
)

FPATH_SYNSET_WORDS = os.path.join(
    DIRPATH_IMAGENET, 'synset_lists', 'synset_words.txt'
)

N_PROCESSES = multiprocessing.cpu_count() // 2


def add_synset_descriptions(df_training_set):
    """Add the synset description to each training image

    :param df_training_set: rows hold a filepath and sysnet ID for a single
     image
    :type df_training_set: pandas.DataFrame
    :return: df_training_set with a 'synset_description' column added
    :rtype: pandas.DataFrame
    """

    synsets = []
    for line in open(FPATH_SYNSET_WORDS):
        synset_id, synset_description = line.strip().split(' ', 1)
        synsets.append({
            'synset': synset_id,
            'synset_description': synset_description
        })
    df_synsets = pd.DataFrame(synsets)
    df_training_set = pd.merge(df_training_set, df_synsets, on='synset')

    return df_training_set


def plot_synset(df_training_set, synset, n_images, dirpath_output):
    """Plot `n_images` from `df_training_set` for the given `synset`

    This will save `n_images` to the `--dirpath_output` passed into the
    script, which defaults to DIRPATH_CLASSIFICATION_PLOTS.

    :param df_training_set: rows hold a filepath and sysnet ID for a single
     image
    :type df_training_set: pandas.DataFrame
    :param synset: identifier of the synset to plot images for
    :type synset: str
    :param n_images: number of images to plot
    :type n_images: int
    :param dirpath_output: directory path to save the images in
    :type dirpath_output: str
    """

    idx_synset = df_training_set['synset'] == synset
    df_synset = df_training_set[idx_synset]
    synset_description = df_synset['synset_description'].iloc[0]

    for fpath_image in df_synset.iloc[:n_images]['fpath_image']:
        fname_image = os.path.basename(fpath_image)
        dirname_image = os.path.basename(os.path.dirname(fpath_image))

        dirpath_image = os.path.join(dirpath_output, dirname_image)
        if not os.path.exists(dirpath_image):
            os.makedirs(dirpath_image)
        fpath_plot = os.path.join(dirpath_image, fname_image)
        image = imageio.imread(fpath_image)

        _, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(synset_description)
        plt.savefig(fpath_plot, bbox_inches='tight')

        plt.clf()
        plt.close()


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    image_classes_arg = parser.add_mutually_exclusive_group()
    image_classes_arg.add_argument(
        '--synset_ids', type=str, nargs='+',
        help='synset IDs to plot images for. By default, the -all_synset_ids '
             'flag is True to plot images for all classes.'
    )
    image_classes_arg.add_argument(
        '--all_synset_ids', action='store_true',
        help='If True, plot images for all classes. Default is True.'
    )

    parser.add_argument(
        '--dirpath_output', type=str, default=DIRPATH_CLASSIFICATION_PLOTS,
        help=(
            'Directory path to save plots in. Defaults to '
            '{}'.format(DIRPATH_CLASSIFICATION_PLOTS)
        )
    )
    parser.add_argument(
        '--from_urls', action='store_true',
        help=(
            'If True, plot images from the training set of images downloaded '
            'from URLS. If False, plot images from the training set of images '
            'downloaded from ImageNet access links.'
        )
    )
            
    parser.add_argument(
        '--n_images', type=int, default=10,
        help='Number of images to plot per class. Defaults to 10.'
    )
    parser.add_argument(
        '--n_processes', type=int, default=N_PROCESSES,
        help=(
            'Number of processes to use to plot images, defaults to {}.'
        ).format(N_PROCESSES)
    )

    args = parser.parse_args()

    if not args.all_synset_ids and not args.synset_ids:
        args.all_synset_ids = True

    return args


def main():
    """Main logic"""

    args = parse_args()

    if not os.path.exists(args.dirpath_output):
        os.makedirs(args.dirpath_output)

    if args.from_urls:
        fpath_df_training_set = os.path.join(
            DIRPATH_IMAGENET, 'from_urls', 'metadata_lists',
            'df_train_set.csv'
        )
    else:
        msg = 'Not passing the --from_urls flag is not supported at this time.'
        raise ValueError(msg)

        fpath_df_training_set = os.path.join(
            DIRPATH_IMAGENET, 'from_access_links', 'metadata_lists',
            'df_train_set.csv'
        )

    df_training_set = pd.read_csv(fpath_df_training_set)
    df_training_set = add_synset_descriptions(df_training_set)
    # ensure plotted images are always the same for a given set of command line
    # arguments
    df_training_set.sort_values('fpath_image', inplace=True)

    if args.synset_ids:
        idx_requested_synsets = df_training_set['synset'].isin(args.synset_ids)
        df_training_set = df_training_set[idx_requested_synsets]

    unique_synsets = df_training_set['synset'].unique()
    with ProcessPoolExecutor(max_workers=args.n_processes) as process_pool:
        futures = [
            process_pool.submit(
                plot_synset, df_training_set, synset, args.n_images,
                args.dirpath_output
            )
            for synset in unique_synsets
        ]

        it = tqdm(as_completed(futures), total=len(unique_synsets))
        for _ in it:
            pass


if __name__ == '__main__':
    main()
