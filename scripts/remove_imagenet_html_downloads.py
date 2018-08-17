#!/usr/bin/env python
"""Remove image URLs that were downloaded as HTML files and not JPGs

This assumes that the following command (or a similar one adjusted for
differences in the paths images were stored at) was run from the terminal:

    find /data/imagenet/images -type f | xargs file | grep HTML | \
        tee /data/imagenet/metadata_lists/html_files.txt | wc -l

This command will find all downloaded image URLs that were saved as HTML files
rather than JPG files, i.e. were corrupted in some way.
"""

import os

import pandas as pd

from utils import dev_env


DIRPATH_DATA = dev_env.get('imagenet', 'dirpath_data')
DIRPATH_SET_CSVS = os.path.join(DIRPATH_DATA, 'metadata_lists')
FPATH_HTML_FILES_TXT = os.path.join(
    DIRPATH_DATA, 'metadata_lists', 'html_files.txt'
)


def remove_html_files(html_fnames, set_name):
    """Load and remove the html filenames from the df_{set_name}_set.csv

    This overwrites the original CSV after it removes the html filenames.

    :param html_fnames: holds the filenames of the HTML files to remove
    :type html_fnames: list[str]
    :param set_name: set of the dataset split to remove html files from, one of
     'train', 'val', or 'test'
    :type set_name: str
    """

    fname_csv = 'df_{}_set.csv'.format(set_name)
    fpath_csv = os.path.join(DIRPATH_SET_CSVS, fname_csv)
    df_set = pd.read_csv(fpath_csv)

    df_set['fname_image'] = df_set['fpath_image'].apply(os.path.basename)
    idx_remove = df_set['fname_image'].isin(html_fnames)
    msg = (
        'Removing {} images from df_{}_set.csv'
    ).format(idx_remove.sum(), set_name)
    print(msg)

    df_set = df_set[~idx_remove]
    df_set.to_csv(fpath_csv, index=False)


def main():
    """Main logic"""

    if not os.path.exists(FPATH_HTML_FILES_TXT):
        msg = (
            '{} doesn\'t exit, and it\'s necessary for removing the image '
            'URLs that were downloaded as HTML files instead of JPGs. To '
            'generate this file, type the following at the command line: '
            'find /data/imagenet/images -type f | xargs file | grep HTML '
            ' | tee /data/imagenet/metadata_lists.html_files.txt | wc -l '
            '\n\n'
            '**Note**: This command may need to be updated if your images are '
            'stored at a different location, and this may take 12+ hours to '
            'run depending on how many images you have stored.'
        ).format(FPATH_HTML_FILES_TXT)
        raise FileNotFoundError(msg)

    df_html_files = pd.read_table(FPATH_HTML_FILES_TXT, names=['text'])
    df_image_ids, _ = df_html_files['text'].str.split(' ', 1).str
    # this turns '/n01367772/n01367772_3576:' => 'n01367772_3576'
    df_html_fnames = df_image_ids.apply(
        lambda text: text.split('/')[-1].replace(':', '')
    )

    for set_name in ['train', 'val', 'test']:
        remove_html_files(df_html_fnames.values, set_name)


if __name__ == '__main__':
    main()
