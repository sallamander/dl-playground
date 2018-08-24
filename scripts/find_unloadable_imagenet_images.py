#!/usr/bin/env python
r"""Find images that are unloadable

This finds images that can't be loaded by `imageio`, which should act as a good
proxy for finding images that were downloaded incorrectly and / or corrupt. It
also flags images that are not 3D, as these images are likely the flicker
"banner" images that simly state "image is no longer available" (i.e. 2D), or
not desirable for training.
"""

from argparse import ArgumentParser
from concurrent.futures import as_completed, ThreadPoolExecutor
import os
import warnings

import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import dev_env


DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_IMAGES = os.path.join(DIRPATH_DATA, 'images')
DIRPATH_METADATA_LISTS = os.path.join(DIRPATH_DATA, 'metadata_lists')

N_THREADS = 10


def get_fpaths_images(synsets):
    """Return a DataFrame holding filepaths of images from `synsets`

    This will find all downloaded images of each synset in `synsets` and return
    a DataFrame holding the absolute filepath as well as the synset ID.

    :param synsets: IDs of all the synsets to get filepaths for
    :type synsets: list[str]
    :return: DataFrame with columns:
    - str synset: ID of the synset
    - str fpath_image: absolute filepath to the image
    :rtype: pandas.DataFrame
    """

    synset_dfs = []
    for synset in synsets:
        dirpath_synset = os.path.join(DIRPATH_IMAGES, synset)
        if not os.path.exists(dirpath_synset):
            continue

        fpath_dicts = []
        for filename in os.listdir(dirpath_synset):
            fpath_image = os.path.join(dirpath_synset, filename)
            fpath_dicts.append({
                'synset': synset,
                'fpath_image': fpath_image
            })

        df_synset_images = pd.DataFrame(fpath_dicts)
        synset_dfs.append(df_synset_images)

    df_fpaths_images = pd.concat(synset_dfs)
    return df_fpaths_images


def load_image(fpath_image):
    """Load the image

    This loads the image to test if (a) it was loaded successfully (i.e. wasn't
    corrupted in download), and (b) ensure that it is not a "flickr" banner
    image that simply states "photo is no longer available." These images come
    as 2D, and thus can be filtered out solely by their number of dimensions.

    :param fpath_image: filepath to the image to load
    :type fpath_image: str
    :return: tuple holding
    - str fpath_image: filepath to the image
    - bool success: whether or not the image was successfully loaded
    :rtype: tuple
    """

    try:
        image = np.array(imageio.imread(fpath_image)).astype(np.float32)
        if image.ndim != 3:
            return fpath_image, False
        return fpath_image, True
    except:
        return fpath_image, False


def find_unloadable_images(fpaths_images, n_threads):
    """Remove images that can't be loaded from fpaths_images

    :param set_name: set of the dataset split to remove bad files from, one of
     'train', 'val', or 'test'
    :type set_name: str
    :param fpaths_images: holds the image filepaths for the given `set_name`
    :type fpaths_images: list[str]
    :param n_threads: number of threads to parallelize loading of images over
    :type n_threads: int
    :return: list of filepaths that couldn't be successfully loaded
    :rtype: list[str]
    """

    with ThreadPoolExecutor(max_workers=n_threads) as thread_pool:
        futures = [
            thread_pool.submit(load_image, fpath_image)
            for fpath_image in tqdm(fpaths_images, total=len(fpaths_images))
        ]

        fpaths_unloadable_images = []
        fpaths_loadable_images = []
        it = tqdm(as_completed(futures), total=len(fpaths_images))
        for future in it:
            fpath_image, success = future.result()
            if success:
                fpaths_loadable_images.append(fpath_image)
            else:
                fpaths_unloadable_images.append(fpath_image)

    total_loaded = len(fpaths_loadable_images) + len(fpaths_unloadable_images)
    assert total_loaded == len(fpaths_images)

    return fpaths_unloadable_images


def parse_args():
    """Parse command line arguments"""

    argparse = ArgumentParser()
    argparse.add_argument(
        '--n_threads', type=int, default=N_THREADS,
        help=(
            'Number of threads to use when laoding images to determine if '
            'they can successfully be loaded, defaults to {}.'
        ).format(N_THREADS)
    )

    args = argparse.parse_args()
    return args


def main():
    """Main logic"""

    args = parse_args()

    fpath_synsets_csv = os.path.join(
        DIRPATH_METADATA_LISTS, 'synset_words.csv'
    )
    df_synsets = pd.read_csv(fpath_synsets_csv)
    df_fpath_images = get_fpaths_images(df_synsets['synset'])

    # raise warnings to play it safe and throw out any errors that might cause
    # problems
    warnings.filterwarnings("error")
    fpaths_images = df_fpath_images['fpath_image'].tolist()
    fpaths_unloadable_images = find_unloadable_images(
        fpaths_images, args.n_threads
    )

    fpath_unloadable_images = os.path.join(
        DIRPATH_METADATA_LISTS, 'unloadable_images.csv'
    )
    np.savetxt(
        fpath_unloadable_images, fpaths_unloadable_images,
        delimiter=',', fmt='%s', header='fpath_image',
        comments=''
    )

if __name__ == '__main__':
    main()
