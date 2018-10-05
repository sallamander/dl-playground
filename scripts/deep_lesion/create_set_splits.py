#! /usr/bin/env python
"""Split the DeepLesion dataset into train, test, and val sets

Note that this script expects a metadata CSV ('DL_info.csv') script to be
present at '/data/deep_lesion/metadata_lists/DL_info.csv'. This CSV contains
pre-defined train, test, and val splits that will be used for training, along
with other relevant metadata. It must be downloaded manually from [1], and
further information on the CSVs contents can be found in the 'readme.pdf' file
at [1].

[1] Dropbox source: https://nihcc.app.box.com/v/DeepLesion
"""

import os

import numpy as np
import pandas as pd

from utils import dev_env

DIRPATH_DATA = dev_env.get('deep_lesion', 'dirpath_data')
FPATH_METADATA_CSV = os.path.join(
    DIRPATH_DATA, 'metadata_lists', 'DL_info.csv'
)

FPATH_TRAIN_SET = os.path.join(
    DIRPATH_DATA, 'metadata_lists', 'df_train_set.csv'
)
FPATH_VAL_SET = os.path.join(
    DIRPATH_DATA, 'metadata_lists', 'df_val_set.csv'
)
FPATH_TEST_SET = os.path.join(
    DIRPATH_DATA, 'metadata_lists', 'df_test_set.csv'
)


def aggregate_duplicate_slices(df_metadata):
    """Aggregate the duplicate slices in `df_metadata`

    Duplicate slices can be identified as those that have the same
    `fname_image` value. All of the column values for these duplicate rows are
    the same, with the exception of `bounding_boxes`, which needs to be
    combined. The result of this will be that `df_metadata` will have one row
    per slice, with all bounding boxes for that slice stored in the same array.

    :param df_metadata: holds metadata for all slices and lesions, with some
     duplicated slices that have multiple bounding boxes
    :type df_metadata: pandas.DataFrame
    :return: metadata for all slices and lesions, with *one row* per slice
    :rtype: pandas.DataFrame
    """

    fname_groupby = df_metadata[
        ['fname_image', 'bounding_boxes']
    ].groupby('fname_image')
    bounding_boxes = fname_groupby['bounding_boxes'].apply(np.hstack)
    df_bounding_boxes = pd.DataFrame(bounding_boxes)

    df_metadata.drop('bounding_boxes', axis=1, inplace=True)
    df_metadata.drop_duplicates('fname_image', inplace=True)

    df_merged = pd.merge(
        df_metadata, df_bounding_boxes, on='fname_image'
    )
    return df_merged


def load_df_metadata():
    """Load the metadata dataframe from FPATH_METADATA_CSV

    :return: dataframe holding the loaded metadata, with columns filtered,
     renamed, and cleaned
    :rtype: pandas.DataFrame
    """

    df_metadata = pd.read_csv(FPATH_METADATA_CSV)

    column_renames = {
        'File_name': 'fname_image',
        'Patient_index': 'patient_id',
        'Study_index': 'study_id',
        'Series_ID': 'series_id',
        'Bounding_boxes': 'bounding_boxes',
        'Coarse_lesion_type': 'lesion_type',
        'Possibly_noisy': 'noisy',
        'Slice_range': 'context_slices',
        'Spacing_mm_px_': 'pixel_spacing',
        'DICOM_windows': 'dicom_windows',
        'Train_Val_Test': 'dataset_split'
    }
    df_metadata = df_metadata[list(column_renames.keys())]
    df_metadata.rename(columns=column_renames, inplace=True)

    def arrayify(string):
        """Turn the provided string into an array of its elements

        I.e. '226.169, 90.0204' => np.array([226.169, 90.0204])

        :param string: string to arrayify
        :type string: str
        :return: array of the strings comma separate elements
        :rtype: numpy.ndarray
        """

        elements = [float(element) for element in string.split(',')]
        return np.array(elements)

    arrayify_columns = [
        'bounding_boxes', 'pixel_spacing', 'context_slices', 'dicom_windows'
    ]
    for column in arrayify_columns:
        df_metadata[column] = df_metadata[column].apply(arrayify)

    return df_metadata


def validate_splits(df_train, df_val, df_test):
    """Check that that are no patient IDs that span sets

    :param df_train: holds metadata for the training set
    :type df_train: pandas.DataFrame
    :param df_val: holds metadata for the validation set
    :type df_val: pandas.DataFrame
    :param df_test: holds metadata for the test set
    :type df_test: pandas.DataFrame
    :raises: AssertionError
    """

    train_patients = set(df_train['patient_id'])
    val_patients = set(df_val['patient_id'])
    test_patients = set(df_test['patient_id'])

    msg = 'train set patients overlap val set patients'
    assert not train_patients.intersection(val_patients), msg

    msg = 'train set patients overlap test set patients'
    assert not train_patients.intersection(test_patients), msg

    msg = 'val set patients overlap test set patients'
    assert not val_patients.intersection(test_patients), msg


def main():
    """Main"""

    df_metadata = load_df_metadata()
    df_metadata = aggregate_duplicate_slices(df_metadata)

    # per the DeepLesion documentation, 1 is the training set, 2 is the
    # validation set, and 3 is the test set
    idx_train = df_metadata['dataset_split'] == 1
    idx_val = df_metadata['dataset_split'] == 2
    idx_test = df_metadata['dataset_split'] == 3

    df_train = df_metadata[idx_train]
    df_val = df_metadata[idx_val]
    df_test = df_metadata[idx_test]

    validate_splits(df_train, df_val, df_test)

    df_train.drop('dataset_split', axis=1, inplace=True)
    df_train.to_csv(FPATH_TRAIN_SET, index=False)
    df_val.drop('dataset_split', axis=1, inplace=True)
    df_val.to_csv(FPATH_VAL_SET, index=False)
    df_test.drop('dataset_split', axis=1, inplace=True)
    df_test.to_csv(FPATH_TEST_SET, index=False)


if __name__ == '__main__':
    main()
