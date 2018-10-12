#! /usr/bin/env python
"""Plot images and bbox annotations from the provided `df_fpaths`"""

import argparse
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
import multiprocessing
import os
import xml.etree.ElementTree as ET

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


N_PROCESSES = multiprocessing.cpu_count() // 2


def parse_image_and_bboxes(fpath_image, fpath_xml):
    """Return the image and bounding boxes from the provided filepaths

    :param fpath_image: filepath to the image to parse
    :type fpath_image: str
    :param fpath_xml: filepath to the XML containing the bounding boxes for
     the image at `fpath_image`
    :type fpath_xml: str
    :return: tuple of
    - numpy.ndarray image: image data from `fpath_image`
    - dict bboxes: dictionary where keys are strings representing the
      synset ID and values are lists of bounding boxes, of shape
      [xmin, ymin, xmax, ymax]
    """

    xml_tree = ET.parse(fpath_xml)
    bboxes_xml = list(xml_tree.getroot().findall('./object/bndbox'))
    bbox_synsets_xml = list(xml_tree.getroot().findall('.object/name'))

    bboxes = defaultdict(list)
    for bbox_xml, bbox_synset_xml in zip(bboxes_xml, bbox_synsets_xml):
        bbox_as_dict = {}
        for child in bbox_xml.getchildren():
            bbox_as_dict[child.tag] = int(child.text)
        bbox = [
            bbox_as_dict['xmin'], bbox_as_dict['ymin'],
            bbox_as_dict['xmax'], bbox_as_dict['ymax']
        ]

        synset = list(bbox_synset_xml.itertext())[0]
        bboxes[synset].append(bbox)

    image = imageio.imread(fpath_image)

    return image, bboxes


def plot_image_and_bboxes(fpath_image, fpath_xml, dirpath_output):
    """Plot the image with the bounding boxes specified in `fpath_xml`

    For each synset ID with bounding box labels in `fpath_imaage`, this will
    save an image to the `dirpath_output/synset_id` with the same filename as
    `os.path.basename(fpath_image)`.

    :param fpath_image: filepath to the image to plot
    :type fpath_image: str
    :param fpath_xml: filepath to the XML containing the bounding boxes for
     the image at `fpath_image`
    :type fpath_xml: str
    :param dirpath_output: directory path to save the plots in
    :type dirpath_output: str
    """

    image, bboxes = parse_image_and_bboxes(fpath_image, fpath_xml)

    for synset, bboxes_synset in bboxes.items():
        _, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')

        for bbox_synset in bboxes_synset:
            x_min, y_min, x_max, y_max = bbox_synset
            rectangle = patches.Rectangle(
                xy=(x_min, y_min),
                width=(x_max - x_min),
                height=(y_max - y_min),
                linewidth=1.5,
                fill=False,
                color='r'
            )

            ax.add_patch(rectangle)

        fname_image = os.path.basename(fpath_image)
        fname_plot = '{}'.format(fname_image)

        dirpath_plot = os.path.join(dirpath_output, synset)
        if not os.path.exists(dirpath_plot):
            os.makedirs(dirpath_plot)
        fpath_plot = os.path.join(dirpath_plot, fname_plot)

        plt.savefig(fpath_plot, bbox_inches='tight')


def parse_args():
    """Parse command line arguments

    :return: name space holding the command line arguments
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dirpath_output', type=str, required=True,
        help=('Directory path to save plots in; if it doesn\'t exist, it will '
              'be created.')
    )
    parser.add_argument(
        '--fpath_df_fpaths', type=str, required=True,
        help=(
            'Filepath to a CSV full of image filepaths and bounding bbox '
            'XMLs to plot. The CSV must contain a \'fpath_image\' and '
            '\'fpath_xml\' column that point to the filepath of the image '
            'and its corresponding XML file of bounding boxes.'
        )
    )

    parser.add_argument(
        '--n_images', type=int, default=100,
        help='Number of images with bboxes to plot. Defaults to 100.'
    )
    parser.add_argument(
        '--n_processes', type=int, default=N_PROCESSES,
        help=(
            'Number of processes to use to plot images, defaults to {}.'
        ).format(N_PROCESSES)
    )

    args = parser.parse_args()

    return args


def main():
    """Main logic"""

    args = parse_args()

    if not os.path.exists(args.dirpath_output):
        os.makedirs(args.dirpath_output)

    df_fpaths = pd.read_csv(args.fpath_df_fpaths)
    # ensure plotted images are always the same for a given set of command line
    # arguments
    df_fpaths.sort_values('fpath_image', inplace=True)
    # randomly shuffle the data, but with a fixed seed so the plotted results
    # are always the same
    df_fpaths = df_fpaths.sample(frac=1, random_state=529)
    df_fpaths = df_fpaths[:args.n_images]

    it = df_fpaths[['fpath_image', 'fpath_xml']].values
    with ProcessPoolExecutor(max_workers=args.n_processes) as process_pool:
        futures = [
            process_pool.submit(
                plot_image_and_bboxes, fpath_image, fpath_xml,
                args.dirpath_output
            )
            for fpath_image, fpath_xml in it
        ]

        it = tqdm(as_completed(futures), total=len(df_fpaths))
        for _ in it:
            pass


if __name__ == '__main__':
    main()
