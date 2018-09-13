"""DataSet for training a model on ImageNet"""

import tensorflow as tf

from datasets.ops import load_image, reshape_image_and_label


class ImageNetDataSet(object):
    """ImageNet dataset"""

    def __init__(self, df_images, dataset_config):
        """Init

        dataset_config must contain the following keys:
        - int or float height: height to reshape the images to
        - int or float width: width to reshape the images to
        - int or float batch_size: batch size to use for returning batches

        :param df_images: holds the filepath to the input image ('fpath_image')
         and the target label for the image ('label')
        :type df_images: pandas.DataFrame
        :param dataset_config: specifies the configuration of the dataset
        :type dataset_config: dict
        """

        self._validate_config(dataset_config)

        if set(df_images.columns) < {'fpath_image', 'label'}:
            msg = (
                'df_images must have an \'fpath_image\' and \'label\' '
                'column, and only {} columns were given.'
            ).format(df_images.columns)
            raise KeyError(msg)

        self.df_images = df_images

        self.height = dataset_config['height']
        self.width = dataset_config['width']
        self.batch_size = dataset_config['batch_size']

        self.num_parallel_calls = dataset_config.get('num_parallel_calls', 4)

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.df_images)

    @staticmethod
    def _validate_config(dataset_config):
        """Vaildate that the necessary keys are in the dataset_config

        This raises a KeyError if there are required keys that are missing, and
        otherwise does nothing.

        :param dataset_config: specifies the configuration for the dataset
        :type dataset_config: dict
        """

        required_keys = {'height', 'width', 'batch_size'}
        missing_keys = required_keys - set(dataset_config)

        if missing_keys:
            msg = (
                '{} keys are missing from the dataset_config, but are '
                'required in order to construct the ImageNetDataSet.'
            ).format(missing_keys)
            raise KeyError(msg)

    def get_infinite_iter(self):
        """Return a tf.data.Dataset that iterates over the data indefinitely

        :return: dataset that iterates over the data indefinitely
        :rtype: tensorflow.data.Dataset
        """

        inputs = self.df_images['fpath_image'].values
        outputs = self.df_images['label'].values
        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        dataset = dataset.map(
            load_image, num_parallel_calls=self.num_parallel_calls
        )
        dataset = dataset.shuffle(buffer_size=len(self.df_images))

        target_image_shape = (self.height, self.width)
        dataset = dataset.map(
            lambda image, label: reshape_image_and_label(
                image, label, target_image_shape, num_label_classes=1000
            )
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()

        return dataset
