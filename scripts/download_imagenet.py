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
DIRPATH_IMAGES = os.path.join(DIRPATH_DATA, 'images')
DIRPATH_METADATA_LISTS = os.path.join(DIRPATH_DATA, 'metadata_lists')
FPATH_FAILED_URLS_CSV = os.path.join(
    DIRPATH_METADATA_LISTS, 'failed_download_urls.csv'
)
URLS_LIST_URL = (
    'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
)

N_THREADS_DEFAULT = 100
FAILED_URLS_LOCK = Lock()
FAILED_URLS = []


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
