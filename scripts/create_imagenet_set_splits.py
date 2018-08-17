#! /usr/bin/env python
"""Split Imagnet images into train, val, and test sets"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import dev_env

DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_IMAGES = os.path.join(DIRPATH_DATA, 'images')
DIRPATH_METADATA_LISTS = os.path.join(DIRPATH_DATA, 'metadata_lists')


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


def main():
    """Main logic"""

    fpath_synsets_list = os.path.join(
        DIRPATH_METADATA_LISTS, 'synset_words.txt'
    )
    df_synset_words = pd.read_table(fpath_synsets_list, names=['text'])
    df_synsets, _ = df_synset_words['text'].str.split(' ', 1).str
    synsets = df_synsets.values

    df_fpath_images = get_fpaths_images(synsets)
    df_train, df_test = train_test_split(
        df_fpath_images, train_size=0.80, test_size=0.20, random_state=529,
        stratify=df_fpath_images['synset']
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
