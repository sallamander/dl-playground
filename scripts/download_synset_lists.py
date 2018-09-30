#! /usr/bin/env python
"""Download ImageNet images from the fall11_urls"""

import os
import subprocess

import pandas as pd

from utils import dev_env

DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_SYNSET_LISTS = os.path.join(DIRPATH_DATA, 'synset_lists')
SYNSETS_URL = 'http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'


def download_synset_lists():
    """Download relevant synset lists

    These lists include the synset IDs for the classification and detection
    challenges, including the mapping from ID to word.

    This function downloads a tarred directory containing these lists to
    DIRPATH_SYNSET_LISTS, and then untars it to grab the relevant lists. It
    removes unused lists, and also translates the `det_synset_words.txt` and
    `synset_words.txt` to CSVs for ease of use later.
    """

    fname_tarfile = os.path.basename(SYNSETS_URL)
    fpath_tarfile = os.path.join(DIRPATH_SYNSET_LISTS, fname_tarfile)

    cmd = 'wget {} -P {}'.format(SYNSETS_URL, DIRPATH_SYNSET_LISTS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    cmd = 'tar -xvf {} -C {}'.format(fpath_tarfile, DIRPATH_SYNSET_LISTS)
    process = subprocess.Popen(cmd.split())
    process.communicate()

    fnames_to_remove = [
        'imagenet.bet.pickle', 'imagenet_mean.binaryproto', 'synsets.txt',
        'train.txt', 'val.txt', 'test.txt'
    ]
    for fname in fnames_to_remove:
        os.remove(os.path.join(DIRPATH_SYNSET_LISTS, fname))

    fnames_synset_txts = ['det_synset_words.txt', 'synset_words.txt']
    for fnames_synset_txt in fnames_synset_txts:
        fpath_synset_txt = os.path.join(
            DIRPATH_SYNSET_LISTS, fnames_synset_txt
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


def main():
    """Main logic"""

    download_synset_lists()

if __name__ == '__main__':
    main()
