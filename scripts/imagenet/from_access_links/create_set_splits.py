#! /usr/bin/env python
"""Create train, val, and test set DataFrames for training

This script assumes that ImageNet was downloaded from the access links provided
when access is granted through imagenet-org.com/downloads, and that the images
are stored in a similar format. While the directory paths used throughout the
script can be altered, the script is currently built to work with the
downloaded data in the following directory structure:

/data/imagenet/from_access_links/

|--- ILSVRC
|    |--- Annotations
|    |--- ImageSets
|    |--- Data
|    |    |--- CLS-LOC
|    |    |    |--- train
|    |    |    |    |--- n04552348
|    |    |    |    |--- n04550184
|    |    |    |    |--- ...
|    |    |    |--- val
|    |    |    |    |--- ILSVRC2012_val_00050000.JPEG
|    |    |    |    |--- ILSVRC2012_val_00000001.JPEG
|    |    |    |    |--- ...
|    |    |    |--- test
|    |    |    |    |--- ILSVRC2012_test_00100000.JPEG
|    |    |    |    |--- ILSVRC2012_test_00068512.JPEG
|    |    |    |    |--- ...
|    |--- devkit
|    |    |--- data
|    |    |    |--- map_clsloc.txt
|    |    |    |--- ILSVRC2015_clsloc_validation_ground_truth.txt
|    |    |    |--- ...

It will create a `metadata_lists` directory at
/data/imagenet/from_access_links/ and place the train, test, and val dataframe
CSVs in that directory.
"""

import os

import pandas as pd

from utils import dev_env


DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_ILSVRC = os.path.join(DIRPATH_DATA, 'from_access_links', 'ILSVRC')
DIRPATH_METADATA = os.path.join(
    DIRPATH_DATA, 'from_access_links', 'metadata_lists'
)

DIRPATH_IMAGES = os.path.join(DIRPATH_ILSVRC, 'Data', 'CLS-LOC')
DIRPATH_TRAIN_IMAGES = os.path.join(DIRPATH_IMAGES, 'train')
DIRPATH_VAL_IMAGES = os.path.join(DIRPATH_IMAGES, 'val')
DIRPATH_TEST_IMAGES = os.path.join(DIRPATH_IMAGES, 'test')

DIRPATH_DEVKIT = os.path.join(DIRPATH_ILSVRC, 'devkit', 'data')
FPATH_SYNSETS_MAPPING = os.path.join(DIRPATH_DEVKIT, 'map_clsloc.txt')
FPATH_SYNSETS_VAL = os.path.join(
    DIRPATH_DEVKIT, 'ILSVRC2015_clsloc_validation_ground_truth.txt'
)


def add_train_set_labels(df_fpaths_images, df_synsets_mapping):
    """Merge on the synset label for each image in the training set

    The training set image labels are obtained by parsing the wordnet synset ID
    from the filename, and then merging on the number label for that particlar
    synset ID, e.g. 'n01440764_10026.JPEG' => 'n01440764' => 449 ('tench').

    :param df_fpaths_images: holds the image filepaths and filenames for the
     images in the training set
    :type df_fpaths_images: pandas.DataFrame
    :param df_synsets_mapping: holds the mapping from wordnet synset ID to the
     numerical label assigned to that wordnet synset ID
    :type df_synsets_mapping: pandas.DataFrame
    :return: df_fpaths_images with a 'label' and 'synset' column added
    :rtype: pandas.DataFrame
    """

    df_fpaths_images['synset'] = df_fpaths_images['fname_image'].apply(
        lambda fname_image: fname_image.split('_')[0]
    )
    df_fpaths_images = pd.merge(
        df_fpaths_images, df_synsets_mapping, on='synset'
    )

    return df_fpaths_images

def add_val_set_labels(df_fpaths_images, df_synsets_mapping):
    """Merge on the synset labels for each image in the validation set

    The validation set image labels are obtained by parsing the filepath for
    the image number (e.g. 'ILSVRC2012_val_00049991' => 49991) and then merging
    on the synset label for the corresponding index in FPATH_SYNSETS_VAL. This
    file simply contains a list of synset labels for each image in the
    validation set, where the first row contains the label for image 1, the
    second row for image 2, etc.

    :param df_fpaths_images: holds the image filepaths and filenames for the
     images in the validation set
    :type df_fpaths_images: pandas.DataFrame
    :param df_synsets_mapping: holds the mapping from wordnet synset ID and the
     numerical label assigned to that wordnet synset ID
    :type df_synsets_mapping: pandas.DataFrame
    :return: df_fpaths_images with a 'label' and 'synset' column added
    :rtype: pandas.DataFrame
    """

    df_fpaths_images['image_number'] = df_fpaths_images['fname_image'].apply(
        lambda fname_image: int(fname_image.split('_')[-1].split('.')[0])
    )

    df_synsets_val = pd.read_table(FPATH_SYNSETS_VAL, names=['label'])
    # add one because 'image_number' starts at 1, not 0
    df_synsets_val.index = df_synsets_val.index + 1
    df_fpaths_images_labeled = pd.merge(
        df_fpaths_images, df_synsets_val,
        left_on='image_number', right_index=True
    )

    assert (
        df_fpaths_images['image_number'].nunique() ==
        df_fpaths_images_labeled['image_number'].nunique()
    )
    assert len(df_fpaths_images_labeled) == len(df_fpaths_images)
    assert df_fpaths_images_labeled['label'].nunique() == 1000

    df_fpaths_images_labeled = pd.merge(
        df_fpaths_images_labeled, df_synsets_mapping, on='label'
    )
    df_fpaths_images_labeled.drop(columns='image_number', inplace=True, axis=1)
    return df_fpaths_images_labeled


def get_fpaths_images(dirpath_images):
    """Return a DataFrame holding filepaths of images from `dirpath_images`

    :param dirpath_images: directory path full of images
    :type dirpath_images: str
    :return: DataFrame with columns:
    - str fpath_image: absolute filepath to the image
    - str fname_image: filename of the image
    """

    fpaths_images = []
    for dirpath, dirnames, fnames in os.walk(dirpath_images, topdown=False):
        for fname_image in fnames:
            fpath_image = os.path.join(dirpath, fname_image)
            fpaths_images.append({
                'fpath_image': fpath_image,
                'fname_image': fname_image
            })

    df_fpaths_images = pd.DataFrame(fpaths_images)
    return df_fpaths_images


def main():
    """Main"""

    if not os.path.exists(DIRPATH_METADATA):
        os.makedirs(DIRPATH_METADATA, exist_ok=True)

    sets = {
        'train': {'dirpath_images': DIRPATH_TRAIN_IMAGES},
        'val': {'dirpath_images': DIRPATH_VAL_IMAGES},
        'test': {'dirpath_images': DIRPATH_TEST_IMAGES}
    }

    df_synsets_mapping = pd.read_table(
        FPATH_SYNSETS_MAPPING, names=['synset', 'label', 'description'],
        sep=' '
    )
    df_synsets_mapping.drop(columns=['description'], inplace=True, axis=1)

    for set_name, set_metadata in sets.items():
        dirpath_images = set_metadata['dirpath_images']
        df_fpaths_images = get_fpaths_images(dirpath_images)

        if set_name == 'train':
            df_fpaths_images = add_train_set_labels(
                df_fpaths_images, df_synsets_mapping
            )
        elif set_name == 'val':
            df_fpaths_images = add_val_set_labels(
                df_fpaths_images, df_synsets_mapping
            )

        fpath_df_fpaths_images = os.path.join(
            DIRPATH_METADATA, 'df_{}_set.csv'.format(set_name)
        )
        df_fpaths_images.to_csv(fpath_df_fpaths_images, index=False)


if __name__ == '__main__':
    main()
