"""Base numpy-backed dataset from which all datasets inherit"""

from abc import abstractproperty

from torch.utils.data import Dataset

from utils.generic_utils import validate_config


class NumPyDataset(Dataset):
    """Abstract base numpy-backed dataset"""

    def __init__(self, config):
        """Init

        :param config: specifies the configuration of the dataset
        :type config: dict
        """

        validate_config(config, self.required_config_keys)
        self.config = config

    @abstractproperty
    def input_keys(self):
        """Return the sample keys that denote a learning algorithm's inputs

        These keys should be contained in the dictionary returned from
        __getitem__, and correspond to the keys that will be the inputs to a
        neural network.

        :return: input key names
        :rtype: list[str]
        """
        raise NotImplementedError

    @abstractproperty
    def sample_shapes(self):
        """Return shapes of the sample elements returned from __getitem__

        :return: dict holding tuples of the shapes for the elements of the
         sample returned from __getitem__
        :return: dict{str: tuple}
        """
        raise NotImplementedError

    @abstractproperty
    def required_config_keys(self):
        """Return the keys required to be in the config passed to the __init__

        :return: required configuration keys
        :rtype: set{str}
        """
        raise NotImplementedError

    @abstractproperty
    def sample_types(self):
        """Return data types of the sample elements returned from __getitem__

        :return: element data types for each element in a sample returned from
         __getitem__
        :rtype: dict{str: str}
        """
        raise NotImplementedError

    @abstractproperty
    def target_keys(self):
        """Return the sample keys that denote a learning algorithm's targets

        These keys should be contained in the dictionary returned from
        __getitem__, and correspond to the keys that will be the targets to a
        neural network.

        :return: target key names
        :rtype: list[str]
        """
        raise NotImplementedError
