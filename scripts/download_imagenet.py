#! /usr/bin/env python
"""Download ImageNet images from the fall11_urls"""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock
import os
import socket
import subprocess
from urllib import request

import pandas as pd
from tqdm import tqdm

from utils import dev_env

DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_BBOX_XMLS = os.path.join(DIRPATH_DATA, 'bbox_xmls')
DIRPATH_IMAGES = os.path.join(DIRPATH_DATA, 'images')
DIRPATH_METADATA_LISTS = os.path.join(DIRPATH_DATA, 'metadata_lists')
FPATH_FAILED_URLS_CSV = os.path.join(
    DIRPATH_METADATA_LISTS, 'failed_download_urls.csv'
)

BBOX_XMLS_URL = 'http://www.image-net.org/Annotation/Annotation.tar.gz'
SYNSETS_URL = 'http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'
URLS_LIST_URL = (
    'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
)

N_THREADS_DEFAULT = 100
FAILED_URLS_LOCK = Lock()
FAILED_URLS = []


def download_bbox_xmls():
    """Download the available bounding box annotations

    This downloads a tarfile that contains a number of other tarfiles that hold
    XML files for each image, where the XML files contain bounding box
    parameterizations for objects in the image. The tarfile will be placed at
    DIRPATH_BBOX_XMLS, and then recursively untarred.
    """

    fname_master_tarfile = os.path.basename(BBOX_XMLS_URL)
    fpath_master_tarfile = os.path.join(
        DIRPATH_BBOX_XMLS, fname_master_tarfile
    )

    cmd = 'wget {} -P {}'.format(BBOX_XMLS_URL, DIRPATH_BBOX_XMLS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    cmd = 'tar -xvf {} -C {}'.format(fpath_master_tarfile, DIRPATH_BBOX_XMLS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    for fname_xml_tarfile in os.listdir(DIRPATH_BBOX_XMLS):
        if fname_xml_tarfile == fname_master_tarfile:
            continue
        else:
            fpath_xml_tarfile = os.path.join(
                DIRPATH_BBOX_XMLS, fname_xml_tarfile
            )
            cmd = 'tar -xvf {} -C {}'.format(
                fpath_xml_tarfile, DIRPATH_BBOX_XMLS
            )
            process = subprocess.Popen(cmd.split())
            process.communicate()

            cmd = 'rm {}'.format(fpath_xml_tarfile)
            process = subprocess.Popen(cmd.split())
            process.communicate()

    # all files untar to an "Annotations" directory inside DIRPATH_BBOX_XMLS,
    # but we want them one level higher
    os.system('mv {0}/Annotation/* {0}'.format(DIRPATH_BBOX_XMLS))

    cmd = 'rm -rf {}/Annotation/'.format(DIRPATH_BBOX_XMLS)
    process = subprocess.Popen(cmd.split())
    process.communicate()


def download_image(image_name, image_url):
    """Download the image from `image_url`

    :param image_name: identifier of the image to download
    :type image_name: str
    :param image_url: URL of the image to download
    :type image_url: str
    """

    dirname_image = image_name.split('_')[0]
    dirpath_save = os.path.join(DIRPATH_IMAGES, dirname_image)
    if not os.path.exists(dirpath_save):
        os.makedirs(dirpath_save, exist_ok=True)
    fpath_save = os.path.join(dirpath_save, image_name)

    if os.path.exists(fpath_save):
        return

    try:
        request.urlretrieve(image_url, fpath_save)
    except:
        FAILED_URLS_LOCK.acquire()
        FAILED_URLS.append((image_name, image_url))
        FAILED_URLS_LOCK.release()


def download_images(df_image_urls, n_threads):
    """Download all images in `df_image_urls`

    :param df_image_urls: DataFrame holding image identifiers and URLs
    :type df_image_urls: pandas.DataFrame
    :param n_threads: number of threads to use to download images
    :type n_threads: int
    """

    with ThreadPoolExecutor(max_workers=n_threads) as thread_pool:
        it = df_image_urls.itertuples()
        for _, image_name, image_url in tqdm(it, total=len(df_image_urls)):
            thread_pool.submit(download_image, image_name, image_url)


def download_image_urls_list():
    """Download the image URL list from the ImageNet website

    This downloads the tarfile containing the list to the
    DIRPATH_METADATA_LISTS and untars it. If the untarred file already exists,
    this function acts as a no-op and simply returns the filepath to the
    untarred file.

    :return: filepath to the textfile of image URLs
    :rtype: str
    """

    fname_tarfile = os.path.basename(URLS_LIST_URL)
    fpath_tarfile = os.path.join(DIRPATH_METADATA_LISTS, fname_tarfile)
    fpath_urls_txt = fpath_tarfile.replace('tgz', 'txt')
    fpath_urls_txt = fpath_urls_txt.replace('imagenet_', '')

    if os.path.exists(fpath_urls_txt):
        return fpath_urls_txt

    cmd = 'wget {} -P {}'.format(URLS_LIST_URL, DIRPATH_METADATA_LISTS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    cmd = 'tar -xvf {} -C {}'.format(fpath_tarfile, DIRPATH_METADATA_LISTS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    return fpath_urls_txt


def download_synset_lists():
    """Download relevant synset lists

    These lists include the synset IDs for the classification and detection
    challenges, including the mapping from ID to word.

    This function downloads a tarred directory containing these lists to
    DIRPATH_METADATA_LISTS, and then untars it to grab the relevant lists. It
    removes unused lists, and also translates the `det_synset_words.txt` and
    `synset_words.txt` to CSVs for ease of use later.
    """

    fname_tarfile = os.path.basename(SYNSETS_URL)
    fpath_tarfile = os.path.join(DIRPATH_METADATA_LISTS, fname_tarfile)

    cmd = 'wget {} -P {}'.format(SYNSETS_URL, DIRPATH_METADATA_LISTS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    cmd = 'tar -xvf {} -C {}'.format(fpath_tarfile, DIRPATH_METADATA_LISTS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    fnames_to_remove = [
        'imagenet.bet.pickle', 'imagenet_mean.binaryproto', 'synsets.txt',
        'train.txt', 'val.txt', 'test.txt'
    ]
    for fname in fnames_to_remove:
        os.remove(os.path.join(DIRPATH_METADATA_LISTS, fname))

    fnames_synset_txts = ['det_synset_words.txt', 'synset_words.txt']
    for fnames_synset_txt in fnames_synset_txts:
        fpath_synset_txt = os.path.join(
            DIRPATH_METADATA_LISTS, fnames_synset_txt
        )
        df_synset_text = pd.read_table(fpath_synset_txt, names=['text'])
        # splits "n01440764 tench, Tinca tinca" =>
        # ["n01440764", "tench, Tinca Tina"]
        synsets, descriptions = df_synset_text['text'].str.split(' ', 1).str
        synsets.name = 'synset'
        descriptions.name = 'description'
        df_synsets = pd.concat([synsets, descriptions], axis=1)

        fpath_synset_csv = fpath_synset_txt.replace('txt', 'csv')
        df_synsets.to_csv(fpath_synset_csv, index=False)


def parse_args():
    """Parse command line arguments"""

    parser = ArgumentParser()

    parser.add_argument(
        '--from_failed_urls', action='store_true',
        help=(
            'If true, load URLs from {} that previously failed to download '
            'and only attempt to download those.'
        ).format(FPATH_FAILED_URLS_CSV)
    )
    parser.add_argument(
        '--n_threads', type=int, default=N_THREADS_DEFAULT,
        help=(
            'Number of threads to use to download images, '
            'defaults to {}.'.format(N_THREADS_DEFAULT)
        )
    )

    args = parser.parse_args()
    return args


def main():
    """Main logic"""

    args = parse_args()

    for dirpath in [DIRPATH_DATA, DIRPATH_IMAGES]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    download_bbox_xmls()
    download_synset_lists()

    if args.from_failed_urls:
        fpath_urls = FPATH_FAILED_URLS_CSV
        delimiter = ','
        if not os.path.exists(fpath_urls):
            msg = (
                '--from_failed_urls flag was passed, but the {} file '
                'specifying previously failed URLs doesn\'t exist. Perhaps '
                'you need to run this script once to generate the file first!'
            ).format(FPATH_FAILED_URLS_CSV)
            raise FileNotFoundError(msg)
    else:
        fpath_urls = download_image_urls_list()
        delimiter = None

    socket.setdefaulttimeout(5)
    df_image_urls_iterator = pd.read_table(
        fpath_urls, names=['image_name', 'image_url'],
        error_bad_lines=False, encoding='ISO=8859-1',
        delimiter=delimiter, chunksize=10000
    )
    for _, df_image_urls in enumerate(tqdm(df_image_urls_iterator)):
        download_images(df_image_urls, args.n_threads)

        # NOTE: (1) This only works inside a Docker container if you have given
        # permission to the container for /proc/sys/vm/drop_caches, which
        # doesn't happen by default. (2) This doesn't seem to accurately work
        # with subprocess.Popen
        os.system('sudo echo 3 | sudo tee /proc/sys/vm/drop_caches')

    df_failed_urls = pd.DataFrame(
        FAILED_URLS, columns=['image_name', 'image_url']
    )
    df_failed_urls.to_csv(FPATH_FAILED_URLS_CSV, index=False)


if __name__ == '__main__':
    main()
