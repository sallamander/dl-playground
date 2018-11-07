#! /usr/bin/env python
"""Plot images from the provided `df_fpaths_images`"""

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
FPATH_SYNSET_WORDS = os.path.join(
    DIRPATH_IMAGENET, 'synset_lists', 'synset_words.txt'
)

N_PROCESSES = multiprocessing.cpu_count() // 2


def add_synset_descriptions(df_fpaths_images):
    """Add the synset description to each training image

    :param df_fpaths_images: rows hold a filepath and sysnet ID for a single
     image
    :type df_fpaths_images: pandas.DataFrame
    :return: df_fpaths_images with a 'synset_description' column added
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
    df_fpaths_images = pd.merge(df_fpaths_images, df_synsets, on='synset')

    return df_fpaths_images


def plot_synset(df_fpaths_images, synset, n_images, dirpath_output):
    """Plot `n_images` from `df_fpaths_images` for the given `synset`

    This will save `n_images` to the `--dirpath_output` passed into the
    script.

    :param df_fpaths_images: rows hold a filepath and sysnet ID for a single
     image
    :type df_fpaths_images: pandas.DataFrame
    :param synset: identifier of the synset to plot images for
    :type synset: str
    :param n_images: number of images to plot
    :type n_images: int
    :param dirpath_output: directory path to save the images in
    :type dirpath_output: str
    """

    idx_synset = df_fpaths_images['synset'] == synset
    df_synset = df_fpaths_images[idx_synset]
    synset_description = df_synset['synset_description'].iloc[0]

    for fpath_image in df_synset.iloc[:n_images]['fpath_image']:
        fname_image = os.path.basename(fpath_image)

        dirpath_image = os.path.join(dirpath_output, synset)
        os.makedirs(dirpath_image, exist_ok=True)

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
    """Parse command line arguments

    :return: name space holding the command line arguments
    :rtype: argparse.Namespace
    """

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
        '--dirpath_output', type=str, required=True,
        help=('Directory path to save plots in; if it doesn\'t exist, it '
              'will be created.')
    )
    parser.add_argument(
        '--fpath_df_fpaths_images', type=str, required=True,
        help=(
            'Filepath to a CSV full of image filepaths to plot. The CSV must '
            'contain a \'fpath_image\' and \'synset\' column that old the '
            'filepath to the image to plot and the wordnet synset ID category '
            'of the image.'
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

    os.makedirs(args.dirpath_output, exist_ok=True)

    df_fpaths_images = pd.read_csv(args.fpath_df_fpaths_images)
    df_fpaths_images = add_synset_descriptions(df_fpaths_images)
    # ensure plotted images are always the same for a given set of command line
    # arguments
    df_fpaths_images.sort_values('fpath_image', inplace=True)

    if args.synset_ids:
        idx_requested_synsets = (
            df_fpaths_images['synset'].isin(args.synset_ids)
        )
        df_fpaths_images = df_fpaths_images[idx_requested_synsets]

    unique_synsets = df_fpaths_images['synset'].unique()
    with ProcessPoolExecutor(max_workers=args.n_processes) as process_pool:
        futures = [
            process_pool.submit(
                plot_synset, df_fpaths_images, synset, args.n_images,
                args.dirpath_output
            )
            for synset in unique_synsets
        ]

        it = tqdm(as_completed(futures), total=len(unique_synsets))
        for _ in it:
            pass


if __name__ == '__main__':
    main()
