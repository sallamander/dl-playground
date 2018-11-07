#! /usr/bin/env python
"""Download the mpii dataset

Reference: http://human-pose.mpi-inf.mpg.de/#download
"""

import os
import subprocess

from utils import dev_env

BASE_URL = 'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/'
ANNOTATIONS_URL = BASE_URL + 'mpii_human_pose_v1_u12_2.zip'
IMAGES_URL = BASE_URL + 'mpii_human_pose_v1.tar.gz'

DIRPATH_DATA = dev_env.get('mpii', 'dirpath_data')
DIRPATH_ANNOTATIONS = os.path.join(DIRPATH_DATA, 'annotations')
DIRPATH_IMAGES = os.path.join(DIRPATH_DATA, 'images')


def download_annotations():
    """Download the mpii annotations

    The zipfile containing the annotations will be downloaded from
    ANNOTATIONS_URL, saved to DIRPATH_ANNOTATIONS, and unzipped.
    """

    fname_annotations_zipfile = os.path.basename(ANNOTATIONS_URL)
    fpath_annotations_zipfile = os.path.join(
        DIRPATH_ANNOTATIONS, fname_annotations_zipfile
    )

    cmd = 'wget {} -P {}'.format(ANNOTATIONS_URL, DIRPATH_ANNOTATIONS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    cmd = 'unzip -q -o {} -d {}'.format(
        fpath_annotations_zipfile, DIRPATH_ANNOTATIONS
    )
    process = subprocess.Popen(cmd.split())
    process.communicate()


def download_images():
    """Download the mpii images

    The tarfile containing the images will be downloaded from
    IMAGES_URL, saved to DIRPATH_IMAGES, and untarred.
    """

    fname_images_zipfile = os.path.basename(IMAGES_URL)
    fpath_images_zipfile = os.path.join(DIRPATH_IMAGES, fname_images_zipfile)

    cmd = 'wget {} -P {}'.format(IMAGES_URL, DIRPATH_IMAGES)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    cmd = 'tar -xvf {} -C {}'.format(fpath_images_zipfile, DIRPATH_IMAGES)
    process = subprocess.Popen(cmd.split())
    process.communicate()


def main():
    """Main"""

    os.makedirs(DIRPATH_ANNOTATIONS, exist_ok=True)
    os.makedirs(DIRPATH_IMAGES, exist_ok=True)

    download_annotations()
    download_images()


if __name__ == '__main__':
    main()
