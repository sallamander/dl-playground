#! /usr/bin/env python
"""Create train, val, and test training dataframes for object detection"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import dev_env

DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_METADATA_LISTS = os.path.join(
    DIRPATH_DATA, 'from_urls', 'metadata_lists'
)
DIRPATH_XMLS = os.path.join(
    DIRPATH_DATA, 'from_urls', 'bbox_xmls'
)

FPATH_DF_FPATHS_IMAGES_FILTERED = os.path.join(
    DIRPATH_METADATA_LISTS, 'df_fpaths_images_filtered.csv'
)
FPATH_TRAIN_SET = os.path.join(
    DIRPATH_METADATA_LISTS, 'df_detection_train_set.csv'
)
FPATH_VAL_SET = os.path.join(
    DIRPATH_METADATA_LISTS, 'df_detection_val_set.csv'
)
FPATH_TEST_SET = os.path.join(
    DIRPATH_METADATA_LISTS, 'df_detection_test_set.csv'
)

FPATH_SYNSET_WORDS = os.path.join(
    DIRPATH_DATA, 'synset_lists', 'synset_words.txt'
)


def add_fpath_xml_column(df_fpaths_images):
    """Add a column holding the XML file holding bbox coordinates

    If the given image doesn't have a corresponding XML file, an `np.nan` will
    be used as a placeholder; these rows should be filtered later, since they
    won't be useful during training.

    :param df_fpaths_images: holds the filepaths for the imagenet images
    :type df_fpaths_images: pandas.DataFrame
    :return: df_fpaths_images with 'fpath_xml' column added
    :rtype: pandas.DataFrame
    """

    def _get_fpath_xml(fpath_image):
        """Return the `fpath_xml` for the corresponding image

        This returns `np.nan` if there is no corresponding XML.

        :param fpath_image: filepath to the imagenet image
        :type fpath_image: str
        :return: filepath to the XML, or null if there isn't one
        :rtype: str or `np.nan`
        """

        # this takes [outer directory path]/n02093056/n02093056_16088 =>
        # n02093056, which is the wordnet synset ID; the XMLs are stored by the
        # wordnet synset ID
        dirname_xml = os.path.basename(os.path.dirname(fpath_image))
        fname = os.path.basename(fpath_image)
        fname_xml = fname + '.xml'

        fpath_xml = os.path.join(DIRPATH_XMLS, dirname_xml, fname_xml)
        if os.path.exists(fpath_xml):
            return fpath_xml
        else:
            return np.nan

    df_fpaths_images['fpath_xml'] = (
        df_fpaths_images['fpath_image'].apply(_get_fpath_xml)
    )
    return df_fpaths_images


def add_label_column(df_fpaths_images):
    """Add the corresponding 'label' for each 'synset' in `df_fpaths_images`

    This translates the 'synset' column to an integer label for use in
    training, i.e. n01580077 => 17.

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

    df_fpaths_inputs_targets = add_fpath_xml_column(df_fpaths_images)
    df_fpaths_inputs_targets = add_label_column(df_fpaths_inputs_targets)

    # factor out null XMLs *post-splitting*, because otherwise there are
    # typically not enough in some classes to split on; note this assumes that
    # the null XMLs will be randomly distributed with respect to the seed we
    # choose
    df_train, df_test = train_test_split(
        df_fpaths_inputs_targets, train_size=0.80, test_size=0.20,
        random_state=529, stratify=df_fpaths_inputs_targets['synset']
    )
    df_val, df_test = train_test_split(
        df_test, train_size=0.50, test_size=0.50, random_state=529,
        stratify=df_test['synset']
    )

    idx_null_xmls_test = pd.isnull(df_test['fpath_xml'])
    df_test = df_test[idx_null_xmls_test]
    idx_null_xmls_val = pd.isnull(df_val['fpath_xml'])
    df_val = df_val[idx_null_xmls_val]
    idx_null_xmls_train = pd.isnull(df_train['fpath_xml'])
    df_train = df_train[idx_null_xmls_train]

    df_train.to_csv(FPATH_TRAIN_SET, index=False)
    df_val.to_csv(FPATH_VAL_SET, index=False)
    df_test.to_csv(FPATH_TEST_SET, index=False)


if __name__ == '__main__':
    main()
