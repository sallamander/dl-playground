#! /usr/bin/env python
"""Create train, val, and test training dataframes for object detection

This script assumes that ImageNet was downloaded from the access links provided
when access is granted through imagenet-org.com/downloads, and that the images
are stored in a similar format. While the directory paths used throughout the
script can be altered, the script is currently built to work with the
downloaded data in the following directory structure:

/data/imagenet/from_access_links/

|--- ILSVRC
|    |--- Annotations
|    |    |--- DET
|    |    |    |--- train
|    |    |    |    |--- ILSVRC_2013_train
|    |    |    |    |--- ILSVRC_2014_train_0000
|    |    |    |    |--- ...
|    |    |    |--- val
|    |    |    |    |--- ILSVRC2012_val_0021650.xml
|    |    |    |    |--- ILSVRC2012_val_0021651.xml
|    |    |    |    |--- ...
|    |--- ImageSets
|    |    |--- DET
|    |    |    |--- train.txt
|    |    |    |--- val.txt
|    |    |    |--- test.txt
|    |    |    |--- ...
|    |    |    |--- train1.txt
|    |    |    |--- train2.txt
|    |    |    |--- train3.txt
|    |    |    |--- ...
|    |--- Data
|    |    |--- DET
|    |    |    |--- train
|    |    |    |    |--- ILSVRC_2013_train
|    |    |    |    |--- ILSVRC_2014_train_0000
|    |    |    |    |--- ...
|    |    |    |--- val
|    |    |    |    |--- ILSVRC2012_val_0021650.JPEG
|    |    |    |    |--- ILSVRC2012_val_0021651.JPEG
|    |    |    |--- test
|    |    |    |    |--- ILSVRC2012_test_00100000.JPEG
|    |    |    |    |--- ILSVRC2012_test_00068512.JPEG
|    |    |    |    |--- ...
|    |--- devkit
|    |    |--- data
|    |    |    |--- ILSVRC2015_det_validation_blacklist.txt
|    |    |    |--- ...

It will create a `metadata_lists` directory at
/data/imagenet/from_access_links/ if it doesn't exist and place the train,
test, and val dataframe CSVs in that directory.
"""

import os

import numpy as np
import pandas as pd

from utils import dev_env


DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_ILSVRC = os.path.join(DIRPATH_DATA, 'from_access_links', 'ILSVRC')
DIRPATH_METADATA_LISTS = os.path.join(
    DIRPATH_DATA, 'from_access_links', 'metadata_lists'
)

DIRPATH_IMAGES = os.path.join(DIRPATH_ILSVRC, 'Data', 'DET')
DIRPATH_XMLS = os.path.join(DIRPATH_ILSVRC, 'Annotations', 'DET')
DIRPATH_IMAGE_LISTS = os.path.join(DIRPATH_ILSVRC, 'ImageSets', 'DET')


def add_fpath_xml_column(df_fpaths_images, set_name):
    """Add a column holding the XML file holding bbox coordinates

    If the given image doesn't have a corresponding XML file, an `np.nan` will
    be used as a placeholder; these rows should be filtered later, since they
    won't be useful during training.

    :param df_fpaths_images: holds the filepaths for the imagenet images
    :type df_fpaths_images: pandas.DataFrame
    :param set_name: set name that the images are from
    :type set_name: str
    :return: df_fpaths_images with 'fpath_xml' column added
    :rtype: pandas.DataFrame
    """

    msg = '`set_name` must be one of {\'train\', \'val\'}'
    assert set_name in {'train', 'val'}, msg

    dirpath_images = os.path.join(DIRPATH_IMAGES, set_name)
    dirpath_xmls = os.path.join(DIRPATH_XMLS, set_name)

    def _get_fpath_xml(fpath_image, dirpath_images, dirpath_xmls):
        """Return the `fpath_xml` for the corresponding image

        :param fpath_image: filepath to the imagenet image
        :type fpath_image: str
        :param dirpath_images: directory path to the images
        :type dirpath_images: str
        :param dirpath_xmls: directory path to the xmls
        :type dirpath_xmls: str
        :return: filepath to the XML
        :rtype: str
        """

        # this splits the filename from the directory path; the corresponding
        # XML filename can then be found by replacing .JPEG by .xml
        fname_image = fpath_image.split(dirpath_images)[1][1:]
        fname_xml = fname_image.replace('.JPEG', '.xml')
        fpath_xml = os.path.join(dirpath_xmls, fname_xml)

        if os.path.exists(fpath_xml):
            return fpath_xml
        else:
            return np.nan

    df_fpaths_images['fpath_xml'] = (
        df_fpaths_images['fpath_image'].apply(
            lambda fpath_image: _get_fpath_xml(
                fpath_image, dirpath_images, dirpath_xmls
            )
        )
    )
    return df_fpaths_images


def get_fpaths_images(set_name):
    """Return a DataFrame of image filepaths for the provided set name

    :param set_name: set name to return the dataframe for
    :type set_name: str
    :return: dataframe holding:
    - str fpath_image: filepath pointing to the input image
    :rtype: pandas.DataFrame
    """

    msg = '`set_name` must be one of {\'train\', \'val\', \'test\'}'
    assert set_name in {'train', 'val', 'test'}, msg

    dirpath_images = os.path.join(DIRPATH_IMAGES, set_name)

    fpath_fnames_set_txt = os.path.join(
        DIRPATH_IMAGE_LISTS, '{}.txt'.format(set_name)
    )
    df_fnames_images = pd.read_table(
        fpath_fnames_set_txt, names=['fname_image', 'observation_id'],
        sep=' '
    )

    df_fnames_images['fpath_image'] = df_fnames_images['fname_image'].apply(
        lambda fname_image: os.path.join(
            dirpath_images, '{}.JPEG'.format(fname_image)
        )
    )
    fpath_exists = df_fnames_images['fpath_image'].apply(os.path.exists)
    assert all(fpath_exists)

    df_fpaths_images = df_fnames_images[['fpath_image']]
    return df_fpaths_images


def main():
    """Main"""

    if not os.path.exists(DIRPATH_METADATA_LISTS):
        os.makedirs(DIRPATH_METADATA_LISTS, exist_ok=True)

    for set_name in ('train', 'val', 'test'):
        df_fpaths_images = get_fpaths_images(set_name=set_name)

        if set_name in {'train', 'val'}:
            df_fpaths_inputs_targets = add_fpath_xml_column(
                df_fpaths_images, set_name
            )

        fpath_set = os.path.join(
            DIRPATH_METADATA_LISTS, 'df_detection_{}_set'.format(set_name)
        )
        df_fpaths_inputs_targets.to_csv(fpath_set, index=False)


if __name__ == '__main__':
    main()
