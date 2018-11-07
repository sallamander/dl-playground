#! /usr/bin/env python
"""Plot images and the joint annotations from the provided `df_annotations`

For details on the dataset annotations, see
http://human-pose.mpi-inf.mpg.de/#download.
"""

import argparse
import os

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

JOINT_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (10, 11), (11, 12),
    (12, 13), (13, 14), (14, 15)
]


def plot_image(idx_image, df_image_annotations, dirpath_output):
    """Plot the image and its joint annotations

    :param idx_image: index of the image in the dataset; used to create the
     filename the plot is saved to
    :type idx_image: int
    :param df_image_annotations: holds joint annotations for persons in the
     image at `idx_image`
    :type df_image_annotations: pandas.DataFrame
    :param dirpath_output: directory to save plots in
    :type dirpath_output: str
    """

    assert df_image_annotations['fpath_image'].nunique() == 1
    fpath_image = df_image_annotations['fpath_image'].unique()[0]
    image = imageio.imread(fpath_image)

    _, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    it = df_image_annotations.groupby('idx_person')
    for _, df_person_annotations in it:
        head_bbox = df_person_annotations['head_bbox'].values[0]
        x_min, y_min, x_max, y_max = head_bbox
        rectangle = patches.Rectangle(
            xy=(x_min, y_min),
            width=(x_max - x_min),
            height=(y_max - y_min),
            linewidth=1.5,
            fill=False,
            color='r'
        )
        ax.add_patch(rectangle)

        plot_joints(ax, df_person_annotations)

        x, y = df_person_annotations['person_position'].values[0]
        ax.plot(x, y, marker='o', markersize=5, color='red')

    fpath_plot = os.path.join(dirpath_output, str(idx_image))
    plt.savefig(fpath_plot, bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_joints(ax, df_joint_annotations):
    """Plot the joint annotations from `df_joint_annotations` onto `ax`

    :param ax: axis to plot the joints on
    :type ax: matplotlib.axes._subplots.AxesSubplot
    :param df_joint_annotations: holds joint annotations for a single person in
     a given image
    :type df_joint_annotations: pandas.DataFrame
    """

    n_joints = df_joint_annotations['joint_id'].nunique()
    assert len(df_joint_annotations) == n_joints
    df_joint_annotations = df_joint_annotations.set_index('joint_id')

    joints = df_joint_annotations['joint_location'].to_dict()
    for joint_pair in JOINT_PAIRS:
        joint1 = joint_pair[0]
        joint2 = joint_pair[1]

        if joint1 in joints and joint2 in joints:
            x1, y1 = joints[joint1]
            x2, y2 = joints[joint2]

            ax.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=2)


def parse_args():
    """Parse command line arguments

    :return: name space holding command line arguments
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dirpath_output', type=str, required=True,
        help=('Directory path to save plots in; if it doesn\'t exist, it '
              'will be created.')
    )
    parser.add_argument(
        '--fpath_df_annotations', type=str, required=True,
        help=(
            'Filepath to a pickled pandas.DataFrame full of image filepaths '
            'and joint annotations.'
        )
    )

    parser.add_argument(
        '--n_images', type=int, default=100,
        help=(
            'Number of images with joint annotations to plot. Defaults to 100.'
        )
    )

    args = parser.parse_args()
    return args


def main():
    """Main"""

    args = parse_args()
    os.makedirs(args.dirpath_output, exist_ok=True)

    df_annotations = pd.read_pickle(args.fpath_df_annotations)
    # ensure plotted images are always the same for a given set of command line
    # arguments
    df_annotations.sort_values('idx_image', inplace=True)
    image_indices_to_keep = (
        df_annotations['idx_image'].drop_duplicates()[:args.n_images]
    )
    df_annotations = df_annotations[
        df_annotations['idx_image'].isin(image_indices_to_keep)
    ]

    it = tqdm(df_annotations.groupby('idx_image'), total=args.n_images)
    for idx_image, df_image_annotations in it:
        plot_image(idx_image, df_image_annotations, args.dirpath_output)


if __name__ == '__main__':
    main()
