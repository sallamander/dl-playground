"""Numpy backed dataset for training a model on ImageNet"""

import imageio
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader

from utils.generic_utils import validate_config


class ImageNetDataSet(Dataset):
    """ImageNet dataset"""

    input_keys = ['image']
    required_config_keys = {'height', 'width'}
    sample_types = {'image': 'float32', 'label': 'uint8'}
    target_keys = ['label']

    def __init__(self, df_images, dataset_config):
        """Init

        `dataset_config` must contain the following keys:
        - int height: height to reshape the images to
        - int width: width to reshape the images to

        :param dataset_config: specifies the configuration of the dataset
        :type dataset_config: dict
        :param df_images: holds the filepath to the input image ('fpath_image')
         and the target label for the image ('label')
        :type df_images: pandas.DataFrame
        """

        validate_config(dataset_config, self.required_config_keys)

        if set(df_images.columns) < {'fpath_image', 'label'}:
            msg = (
                'df_images must have an \'fpath_image\' and \'label\' '
                'column, and only {} columns were given.'
            ).format(df_images.columns)
            raise KeyError(msg)

        self.df_images = df_images
        self.config = dataset_config
        self.df_images['label'] = (
            self.df_images['label'].astype(self.sample_types['label'])
        )

    def __getitem__(self, idx):
        """Return the image, label pair at index `idx` from `self.df_images`

        :return: dict with keys:
        - numpy.ndarray image: pixel data loaded from the `fpath_image` at
          index `idx` of `self.df_images`
        - int label: class label assigned to the returned image
        :rtype: dict
        """

        fpath_image = self.df_images.loc[idx, 'fpath_image']
        image = imageio.imread(fpath_image)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
            image = np.concatenate((image, image, image), axis=-1)

        n_channels = image.shape[-1]
        target_shape = (
            self.config['height'], self.config['width'], n_channels
        )
        image = resize(image, output_shape=target_shape)
        image = image.astype(self.sample_types['image'])

        label = np.array(self.df_images.loc[idx, 'label'])
        assert label.dtype == self.sample_types['label']

        sample = {'image': image, 'label': label}

        return sample

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.df_images)

    def as_generator(self, shuffle=False):
        """Return a generator that yields the entire dataset once

        This is intended to act as a lightweight wrapper around the
        torch.utils.data.DataLoader class, which allows for shuffling of the
        data without loading it into memory. It purposely removes the added
        batch dimension from the DataLoader such that each element yielded is
        still a single sample, just as if it came from indexing into this
        class, e.g. ImageNetDataSet[10].

        :param shuffle: if True, shuffle the data before returning it
        :type shuffle: bool
        :return: generator that yields the entire dataset once
        :rtype: generator
        """

        for sample in DataLoader(dataset=self, shuffle=shuffle):
            sample_batch_dim_removed = {}
            for key, val in sample.items():
                sample_batch_dim_removed[key] = val[0]
            yield sample_batch_dim_removed
