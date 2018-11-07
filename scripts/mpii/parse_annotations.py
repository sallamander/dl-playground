#! /usr/bin/env python
"""Parse the annotations from the original matlab file into a pickled DataFrame

This script assumes that the annotations have already been downloaded. While
the directory paths used throughout the script can be altered, the script is
currently built to work with the downloaded annotations at the following
location: /data/mpii/annotations/mpii_human_pose_v1_u12_2

Reference Implementations:
https://github.com/princeton-vl/pose-hg-train/blob/master/src/misc/convert_annot.py
https://github.com/princeton-vl/pose-hg-train/blob/master/src/misc/mpii.py
"""

import os

import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from utils import dev_env

DIRPATH_MPII = dev_env.get('mpii', 'dirpath_data')
DIRPATH_ANNOTATIONS = os.path.join(DIRPATH_MPII, 'annotations')
FPATH_ANNOTATIONS_MAT = os.path.join(
    DIRPATH_ANNOTATIONS, 'mpii_human_pose_v1_u12_2',
    'mpii_human_pose_v1_u12_1.mat'
)
FPATH_ANNOTATIONS_PICKLE = os.path.join(
    DIRPATH_ANNOTATIONS, 'mpii_human_pose_v1_u12_2', 'df_annotations.pickle'
)


def parse_annotations(annotations):
    """Parse the provided annotations

    For details on the structure of the annotations, see the 'Annotation
    description' section at http://human-pose.mpi-inf.mpg.de/#download.

    Reference Implementations:
    https://github.com/princeton-vl/pose-hg-train/blob/master/src/misc/convert_annot.py
    https://github.com/princeton-vl/pose-hg-train/blob/master/src/misc/mpii.py

    :param annotations: structured array holding the annotations
    :type annotations: numpy.ndarray
    :return: dataframe of parsed joint annotations
    :rtype: pandas.DataFrame
    """

    training_set_indicator = list(annotations['img_train'][0][0][0])
    n_images = len(training_set_indicator)

    annolist = annotations['annolist'][0][0][0]
    images = list(annolist['image'])
    person_annotations_per_image = list(annolist['annorect'])

    joint_annotations = []
    for idx_image in tqdm(range(n_images), total=n_images):
        training_image = training_set_indicator[idx_image]
        if not training_image:
            continue

        fname_image = images[idx_image][0][0][0][0]
        person_annotations = person_annotations_per_image[idx_image]
        if person_annotations.size:
            person_annotations = person_annotations[0]
        else:
            continue

        for idx_person, person_annotation in enumerate(person_annotations):
            joint_annotations.extend(
                parse_person_annotation(
                    idx_image, idx_person, fname_image, person_annotation
                )
            )

    df_annotations = pd.DataFrame(joint_annotations)
    return df_annotations


def parse_person_annotation(idx_image, idx_person, fname_image,
                            person_annotation):
    """Return the joint annotations for the provided `person_annotation`

    :param idx_image: image index that the `person_annotation` corresponds to
    :type idx_image: int
    :param idx_person: person index in the image that the `person_annotation`
     corresponds to
    :type idx_person: int
    :param fname_image: filename of the image the `person_annotation`
     corresponds to
    :type fname_image: str
    :param person_annotation: structured array containing joint annotations for
     a single person in an image
    :type person_annotation: numpy.ndarray
    :return: list of dictionaries with joint annotation information, with keys:
    - int idx_image: index of the image the `person_annotation` corresponds to
    - int idx_person: person index in the image that the `person_annotation`
      corresponds to
    - str fname_image: filename of the image the `person_annotation`
      corresponds to
    - tuple(float) head_bbox: (xmin, ymin, xmax, ymax) of the coordinates of
      the bounding box surrounding the head of the person that the
      `person_annotation` corresponds to
    - float person_scale: scale of the person in `person_annotation`, with
      respect to a height of 200 pixels
    - tuple(float) person_position: image position of the person in
      `person_annotation`
    - int joint_id: unique ID that maps to the type of joint
    - bool is_visible: denotes if the joint is visible or occluded
    - tuple(float) joint_location: (x, y) coordinate of the joint location in
      the image
    """

    if 'scale' not in person_annotation.dtype.fields:
        return []
    scale = person_annotation['scale']
    if scale[0].size:
        scale = scale[0][0]
    else:
        return []

    x_position = person_annotation['objpos'][0]['x'][0][0][0]
    y_position = person_annotation['objpos'][0]['y'][0][0][0]
    position = (x_position, y_position)

    xmin_head_bbox = person_annotation['x1'][0][0]
    xmax_head_bbox = person_annotation['x2'][0][0]
    ymin_head_bbox = person_annotation['y1'][0][0]
    ymax_head_bbox = person_annotation['y2'][0][0]
    head_bbox = (
        xmin_head_bbox, ymin_head_bbox, xmax_head_bbox, ymax_head_bbox
    )

    joint_annotations = []
    joints = person_annotation['annopoints'][0][0][0][0]
    for joint in joints:
        joint_id = joint['id'][0][0]
        x_joint, y_joint = joint['x'][0][0], joint['y'][0][0]

        if 'is_visible' not in joint.dtype.fields:
            continue
        if joint['is_visible'].size:
            is_visible = bool(joint['is_visible'][0][0])
        else:
            is_visible = True

        joint_annotations.append({
            'idx_image': idx_image,
            'idx_person': idx_person,
            'fname_image': fname_image,
            'head_bbox': head_bbox,
            'person_scale': scale,
            'person_position': position,
            'joint_id': joint_id,
            'is_visible': is_visible,
            'joint_location': (x_joint, y_joint),
        })

    return joint_annotations


def main():
    """Main"""

    annotations = loadmat(FPATH_ANNOTATIONS_MAT)['RELEASE']
    df_annotations = parse_annotations(annotations)

    num_duplicates = (
        df_annotations.duplicated(['idx_image', 'idx_person', 'joint_id'])
    ).sum()
    percent_duplicates = num_duplicates / len(df_annotations)
    assert percent_duplicates <= 0.01

    df_annotations.drop_duplicates(
        ['idx_image', 'idx_person', 'joint_id'], inplace=True
    )
    df_annotations.to_pickle(FPATH_ANNOTATIONS_PICKLE)


if __name__ == '__main__':
    main()
