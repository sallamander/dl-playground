"""Numpy backed dataset for training a model on ImageNet"""

import imageio
import numpy as np
from skimage.transform import resize
from trainet.datasets.base_dataset import NumPyDataset
from trainet.utils.generic_utils import cycle, validate_config


class ImageNetDataset(NumPyDataset):
    """ImageNet dataset"""

    def __init__(self, df_obs, config):
        """Init

        `config` must contain the following keys:
        - int height: height to reshape the images to
        - int width: width to reshape the images to

        :param config: specifies the configuration of the dataset
        :type config: dict
        :param df_obs: holds the filepath to the input image ('fpath_image')
         and the target label for the image ('label')
        :type df_obs: pandas.DataFrame
        """

        super().__init__(config)

        if set(df_obs.columns) < {'fpath_image', 'label'}:
            msg = (
                'df_obs must have an \'fpath_image\' and \'label\' '
                'column, and only {} columns were given.'
            ).format(df_obs.columns)
            raise KeyError(msg)

        self.df_obs = df_obs
        self.config = config
        self.df_obs['label'] = (
            self.df_obs['label'].astype(self.sample_types['label'])
        )

    def __getitem__(self, idx):
        """Return the image, label pair at index `idx` from `self.df_obs`

        Occasionally (< 1e-4% of the time), one of the loaded images has either
        2 channels or 4 channels. If the former, the image is simply stacked
        three times to create a 3 channel input, and if the latter, the first
        three channels of the four are taken.

        :param idx: index into self.df_obs of the observation to return
        :type idx: int
        :return: dict with keys:
        - numpy.ndarray image: pixel data loaded from the `fpath_image` at
          index `idx` of `self.df_obs`
        - int label: class label assigned to the returned image
        :rtype: dict
        """

        fpath_image = self.df_obs.loc[idx, 'fpath_image']
        image = imageio.imread(fpath_image)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
            image = np.concatenate((image, image, image), axis=-1)

        n_channels = image.shape[-1]
        if n_channels == 4:
            n_channels = 3
            image = image[:, :, :3]
        target_shape = (
            self.config['height'], self.config['width'], n_channels
        )
        image = resize(image, output_shape=target_shape)
        image = image.astype(self.sample_types['image'])

        label = np.array(self.df_obs.loc[idx, 'label'])
        assert label.dtype == self.sample_types['label']

        sample = {'image': image, 'label': label}
        return sample

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.df_obs)

    @property
    def input_keys(self):
        """Return the sample keys that denote a learning algorithm's inputs

        These keys should be contained in the dictionary returned from
        __getitem__, and correspond to the keys that will be the inputs to a
        neural network.

        :return: input key names
        :rtype: set{str}
        """
        return ['image']

    @property
    def required_config_keys(self):
        """Return the keys required to be in the config passed to the __init__

        :return: required configuration keys
        :rtype: set{str}
        """
        return {'height', 'width'}

    @property
    def sample_shapes(self):
        """Return the shapes of the outputs returned

        :return: dict holding the tuples of the shapes for the values returned
         when iterating over the dataset
        :rtype: dict{str: tuple}
        """

        height = self.config['height']
        width = self.config['width']

        image_shape = (height, width, 3)
        label_shape = (1000, )

        return {'image': image_shape, 'label': label_shape}

    @property
    def sample_types(self):
        """Return data types of the sample elements returned from __getitem__

        :return: element data types for each element in a sample returned from
         __getitem__
        :rtype: dict{str: str}
        """
        return {'image': 'float32', 'label': 'uint8'}

    @property
    def target_keys(self):
        """Return the sample keys that denote a learning algorithm's targets

        These should be contained in the dictionary returned from __getitem__,
        and correspond to the keys that will be the targets to a neural
        network.

        :return: target key names
        :rtype: list[str]
        """
        return ['label']
