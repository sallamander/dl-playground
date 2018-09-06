#! /usr/bin/env python
"""Split Imagnet images into train, val, and test sets"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import dev_env

DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_METADATA_LISTS = os.path.join(DIRPATH_DATA, 'metadata_lists')

FPATH_BAD_FILES_TXT = os.path.join(DIRPATH_METADATA_LISTS, 'bad_files.txt')
FPATH_DF_FPATHS_IMAGES = os.path.join(
    DIRPATH_METADATA_LISTS, 'df_fpaths_images.csv'
)
FPATH_UNLOADABLE_FILES_CSV = os.path.join(
    DIRPATH_METADATA_LISTS, 'unloadable_files.csv'
)


def remove_bad_files(df_fpaths_images):
    """Load and remove the bad filenames from `df_fpaths_images`

    :param df_fpaths_images: holds the filepaths for the imagenet images
    :type df_fpaths_images: pandas.DataFrame
    :return: df_fpaths_images with the bad filenames removed
    :rtype: pandas.DataFrame
    """

    df_bad_files = pd.read_table(FPATH_BAD_FILES_TXT, names=['text'])
    df_image_ids, _ = df_bad_files['text'].str.split(' ', 1).str
    # this turns '/n01367772/n01367772_3576:' => 'n01367772_3576'
    bad_fnames = df_image_ids.apply(
        lambda text: text.split('/')[-1].replace(':', '')
    ).values

    df_fpaths_images['fname_image'] = (
        df_fpaths_images['fpath_image'].apply(os.path.basename)
    )
    idx_remove = df_fpaths_images['fname_image'].isin(bad_fnames)
    msg = (
        'Removing {} text / empty images from df_fpaths_images.'
    ).format(idx_remove.sum())
    print(msg)

    df_fpaths_images = df_fpaths_images[~idx_remove]
    return df_fpaths_images


def remove_unloadable_files(df_fpaths_images):
    """Load and remove the unloadable filenames from `df_fpaths_images`

    :param df_fpaths_images: holds the filepaths for the imagenet images
    :type df_fpaths_images: pandas.DataFrame
    :return: df_fpaths_images with the unloadable filenames removed
    :rtype: pandas.DataFrame
    """

    df_unloadable_files = pd.read_csv(FPATH_UNLOADABLE_FILES_CSV)
    unloadable_fpaths = df_unloadable_files['fpath_image']

    idx_remove = df_fpaths_images['fpath_image'].isin(unloadable_fpaths)
    msg = (
        'Removing {} unloadable images from df_fpaths_images.'
    ).format(idx_remove.sum())
    print(msg)

    df_fpaths_images = df_fpaths_images[~idx_remove]
    return df_fpaths_images


def main():
    """Main logic"""

    files_exist = (
        os.path.exists(FPATH_BAD_FILES_TXT) and
        os.path.exists(FPATH_UNLOADABLE_FILES_CSV)
    )
    if not files_exist:
        msg = (
            '{0} or {1} don\'t exist, and they are necessary for removing the '
            'image URLs that were downloaded as BAD files instead of images, '
            'as well as the files that are unloadable. To generate {0}, type '
            'the following at the command line: '
            'find /data/imagenet/images -type f | xargs file | grep -v '
            r'"image data" | '
            'tee /data/imagenet/metadata_lists/bad_files.txt | wc -l '
            '\n\n'
            '**Note**: This command may need to be updated if your images are '
            'stored at a different location, and this may take 12+ hours to '
            'run depending on how many images you have stored. \n\n '
            'To generate {1}, run the \'find_unloadable_imagenet_images.py\' '
            'script in the scripts directory.'
        ).format(FPATH_BAD_FILES_TXT, FPATH_UNLOADABLE_FILES_CSV)
        raise FileNotFoundError(msg)

    df_fpaths_images = pd.read_csv(FPATH_DF_FPATHS_IMAGES)
    df_fpaths_images = remove_bad_files(df_fpaths_images)
    df_fpaths_images = remove_unloadable_files(df_fpaths_images)

    df_train, df_test = train_test_split(
        df_fpaths_images, train_size=0.80, test_size=0.20, random_state=529,
        stratify=df_fpaths_images['synset']
    )
    df_val, df_test = train_test_split(
        df_test, train_size=0.50, test_size=0.50, random_state=529,
        stratify=df_test['synset']
    )

    fpath_test_set = os.path.join(
        DIRPATH_METADATA_LISTS, 'df_test_set.csv'
    )
    df_test.to_csv(fpath_test_set, index=False)

    fpath_val_set = os.path.join(
        DIRPATH_METADATA_LISTS, 'df_val_set.csv'
    )
    df_val.to_csv(fpath_val_set, index=False)

    fpath_train_set = os.path.join(
        DIRPATH_METADATA_LISTS, 'df_train_set.csv'
    )
    df_train.to_csv(fpath_train_set, index=False)


if __name__ == '__main__':
    main()
