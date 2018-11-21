"""Numpy backed dataset for training a model on ImageNet"""

import imageio
from torch.utils.data import Dataset


class ImageNetDataSet(Dataset):
    """ImageNet dataset"""

    input_keys = ['image']
    target_keys = ['label']

    def __init__(self, df_images):
        """Init

        :param df_images: holds the filepath to the input image ('fpath_image')
         and the target label for the image ('label')
        :type df_images: pandas.DataFrame
        """

        if set(df_images.columns) < {'fpath_image', 'label'}:
            msg = (
                'df_images must have an \'fpath_image\' and \'label\' '
                'column, and only {} columns were given.'
            ).format(df_images.columns)
            raise KeyError(msg)

        self.df_images = df_images
        self.df_images['label'] = (
            self.df_images['label'].astype(self.sample_types['label'])
        )

    def __getitem__(self, idx):
        """Return the image, label pair at index `idx` from `self.df_images`

        :return: dict with keys:
        - numpy.ndarray image: pixel data loaded from the `fpath_image` at
          index `idx` of `self.df_images`
        - int label: class label assigned to the returned image
        """

        fpath_image = self.df_images.loc[idx, 'fpath_image']
        image = imageio.imread(fpath_image).astype(self.sample_types['image'])
        label = self.df_images.loc[idx, 'label']
        assert label.dtype == self.sample_types['label']

        sample = {'image': image, 'label': label}

        return sample

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.df_images)

    @property
    def sample_types(self):
        """Return the output types corresponding to the outputs of the dataset

        :return: output types
        :rtype: dict with keys:
        - str image: holds the data type of the 'image' key for each sample
          returned from the `__getitem__` method
        - str label: holds the data type of the 'label' key for each sample
          returned from the `__getitem__` method
        """

        return {'image': 'float32', 'label': 'uint8'}

    def as_generator(self):
        """Return a generator that yields the entire dataset once

        :return: generator that yields the entire dataset once
        :rtype: generator
        """

        for idx in range(len(self)):
            yield self[idx]
