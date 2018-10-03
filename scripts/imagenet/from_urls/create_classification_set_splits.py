#! /usr/bin/env python
"""Split Imagnet into train, val, and test sets for object classification"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import dev_env

DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_METADATA_LISTS = os.path.join(
    DIRPATH_DATA, 'from_urls', 'metadata_lists'
)

FPATH_DF_FPATHS_IMAGES_FILTERED = os.path.join(
    DIRPATH_METADATA_LISTS, 'df_fpaths_images_filtered.csv'
)

FPATH_SYNSET_WORDS = os.path.join(
    DIRPATH_DATA, 'synset_lists', 'synset_words.txt'
)


def add_label_column(df_fpaths_images):
    """Add the corresponding 'label' for each 'synset' in `df_fpaths_images`

    This translates the 'synset' column to an integer label for use in
    training, i.e. n01580077 => 'jay'.

    :param df_fpaths_images: holds the filepaths for the imagenet images
    :type df_fpaths_images: pandas.DataFrame
    :return: df_fpaths_images with 'label' column added
    :rtype: pandas.DataFrame
    """

    df_synsets_text = pd.read_table(FPATH_SYNSET_WORDS, names=['text'])
    synsets, _ = df_synsets_text['text'].str.split(' ', 1).str
    df_synsets = pd.DataFrame(synsets)
    df_synsets.rename(columns={'text': 'synset'}, inplace=True)
    df_synsets['label'] = df_synsets.index

    df_fpaths_images = pd.merge(df_fpaths_images, df_synsets, on='synset')
    return df_fpaths_images


def main():
    """Main logic"""

    df_fpaths_images = pd.read_csv(FPATH_DF_FPATHS_IMAGES_FILTERED)
    df_fpaths_images = add_label_column(df_fpaths_images)

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
