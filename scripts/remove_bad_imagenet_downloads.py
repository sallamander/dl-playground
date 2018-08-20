#!/usr/bin/env python
r"""Remove image URLs that were downloaded as text / empty files and not images

This assumes that the following command (or a similar one adjusted for
differences in the paths images were stored at) was run from the terminal:

    find /data/imagenet/images -type f | xargs file | \
        grep -v "image data" | \
        tee /data/imagenet/metadata_lists/bad_files.txt | wc -l

This command will find all downloaded image URLs that were saved as text
or empty files rather than image files, i.e. were corrupted or not downloaded
properly in some way.
"""

from argparse import ArgumentParser
from concurrent.futures import as_completed, ThreadPoolExecutor
import os

import imageio
import pandas as pd
from tqdm import tqdm

from utils import dev_env


DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_SET_CSVS = os.path.join(DIRPATH_DATA, 'metadata_lists')
FPATH_BAD_FILES_TXT = os.path.join(
    DIRPATH_DATA, 'metadata_lists', 'bad_files.txt'
)

N_THREADS = 10


def load_image(fpath_image):
    """Load the image

    :param fpath_image: filepath to the image to load
    :type fpath_image: str
    :return: tuple holding
    - str fpath_image: filepath to the image
    - bool success: whether or not the image was successfully loaded
    :rtype: tuple
    """

    try:
        imageio.imread(fpath_image)
        return fpath_image, True
    except:
        return fpath_image, False


def remove_bad_files(bad_fnames, set_name, df_set):
    """Load and remove the bad filenames from `df_set`

    :param bad_fnames: holds the filenames of the text / empty files to remove
    :type bad_fnames: list[str]
    :param set_name: set of the dataset split to remove bad files from, one of
     'train', 'val', or 'test'
    :type set_name: str
    :param df_set: holds the image filepaths for the given `set_name`
    :type df_set: pandas.DataFrame
    :return: `df_set` with `bad_fnames` removed
    :rtype: pandas.DataFrame
    """

    df_set['fname_image'] = df_set['fpath_image'].apply(os.path.basename)
    idx_remove = df_set['fname_image'].isin(bad_fnames)
    msg = (
        'Removing {} text / empty images from df_{}_set.csv'
    ).format(idx_remove.sum(), set_name)
    print(msg)

    df_set = df_set[~idx_remove]
    return df_set


def remove_unloadable_images(set_name, df_set, n_threads):
    """Remove images that can't be loaded from df_set

    :param set_name: set of the dataset split to remove bad files from, one of
     'train', 'val', or 'test'
    :type set_name: str
    :param df_set: holds the image filepaths for the given `set_name`
    :type df_set: pandas.DataFrame
    :param n_threads: number of threads to parallelize loading of images over
    :type n_threads: int
    :return: `df_set` with unloadable images removed
    :rtype: pandas.DataFrame
    """

    fpaths_unloadable_images = []

    with ThreadPoolExecutor(max_workers=n_threads) as thread_pool:
        futures = [
            thread_pool.submit(load_image, fpath_image)
            for fpath_image in tqdm(df_set['fpath_image'], total=len(df_set))
        ]

        fpaths_unloadable_images = []
        fpaths_loadable_images = []
        it = tqdm(as_completed(futures), total=len(df_set))
        for future in it:
            fpath_image, success = future.result()
            if success:
                fpaths_loadable_images.append(fpath_image)
            else:
                fpaths_unloadable_images.append(fpath_image)

    total_loaded = len(fpaths_loadable_images) + len(fpaths_unloadable_images)
    assert total_loaded == len(df_set)

    idx_remove = df_set['fpath_image'].isin(fpaths_unloadable_images)
    msg = (
        'Removing {} unloadable_images from df_{}_set.csv'
    ).format(idx_remove.sum(), set_name)
    print(msg)

    df_set = df_set[~idx_remove]
    return df_set


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

    if not os.path.exists(FPATH_BAD_FILES_TXT):
        msg = (
            '{} doesn\'t exit, and it\'s necessary for removing the image '
            'URLs that were downloaded as BAD files instead of images. To '
            'generate this file, type the following at the command line: '
            'find /data/imagenet/images -type f | xargs file | grep -v '
            r'"image data" | '
            'tee /data/imagenet/metadata_lists/bad_files.txt | wc -l '
            '\n\n'
            '**Note**: This command may need to be updated if your images are '
            'stored at a different location, and this may take 12+ hours to '
            'run depending on how many images you have stored.'
        ).format(FPATH_BAD_FILES_TXT)
        raise FileNotFoundError(msg)

    df_bad_files = pd.read_table(FPATH_BAD_FILES_TXT, names=['text'])
    df_image_ids, _ = df_bad_files['text'].str.split(' ', 1).str
    # this turns '/n01367772/n01367772_3576:' => 'n01367772_3576'
    df_bad_fnames = df_image_ids.apply(
        lambda text: text.split('/')[-1].replace(':', '')
    )

    for set_name in ['train', 'val', 'test']:
        fname_set_csv = 'df_{}_set.csv'.format(set_name)
        fpath_set_csv = os.path.join(DIRPATH_SET_CSVS, fname_set_csv)
        df_set = pd.read_csv(fpath_set_csv)

        remove_bad_files(df_bad_fnames.values, set_name, df_set)
        remove_unloadable_images(set_name, df_set, args.n_threads)
        df_set.to_csv(fpath_set_csv, index=False)


if __name__ == '__main__':
    main()
