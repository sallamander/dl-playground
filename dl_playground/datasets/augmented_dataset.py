"""Dataset for applying transformations to a NumPyDataset"""

from torch.utils.data import Dataset, DataLoader

from datasets.ops import apply_transformation
from utils.generic_utils import cycle


class AugmentedDataset(Dataset):
    """Dataset with transformations applied"""

    def __init__(self, numpy_dataset, transformations=None):
        """Init

        :param numpy_dataset: dataset that provides samples for training
        :type numpy_dataset: datasets.base_dataset.NumPyDataset
        :param transformations: holds 2 element tuples with the first element
         being a function to apply to the dataset samples and the second
         element being a dictionary of keyword arguments to pass to those
         functions
        :type transformations: list[tuple(function, dict)]
        """
        
        self.numpy_dataset = numpy_dataset
        self.transformations = (
            transformations if not None else []
        )

    def __getattr__(self, attr):
        """Returned the specified attribute from self.numpy_dataset
        
        This method is only called when the attribute is not found in
        self.__dict__, and effectively proxies any attribute search to
        self.numpy_dataset.

        :param attr: attribute to return
        :type attr: str
        :return: attribute value
        :rtype: object
        :raises: AttributeError if the attribute is not found on
         self.numpy_dataset
        """

        if not hasattr(self.numpy_dataset, attr):
            msg = (
                '{} attribute not found on AugmentedDataset class or '
                'its numpy_dataset attribute.'
            ).format(attr)
            raise AttributeError(msg)
        else:
            return getattr(self.numpy_dataset, attr)

    def __getitem__(self, idx):
        """Return the transformed sample at index `idx` of `self.numpy_dataset`

        :param idx: the index of the observation to return
        :type idx: int
        :return: sample returned from `self.numpy_dataset.__getitem__`
        :rtype: dict
        """

        sample = self.numpy_dataset[idx]

        it = self.transformations
        for transformation_fn, transformation_fn_kwargs in it:
            transformation_fn_kwargs = transformation_fn_kwargs.copy()
            sample_keys = transformation_fn_kwargs.pop('sample_keys')

            sample = apply_transformation(
                transformation_fn, sample, sample_keys,
                transformation_fn_kwargs
            )

        return sample

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.numpy_dataset)
    
    def as_generator(self, shuffle=False, n_workers=0):
        """Return a generator that yields the entire dataset once

        This method is intended to act as a lightweight wrapper around the
        torch.utils.data.DataLoader class, which has built-in shuffling of the
        data without loading it all into memory. This method purposely removes
        the added batch dimension from DataLoader such that each element
        yielded is still a single sample, just as if it came from indexing into
        this class, e.g. AugmentedDataset[10].
        
        :param shuffle: if True, shuffle the data before returning it
        :type shuffle: bool
        :param n_workers: number of subprocesses to use for data loading
        :type n_workers: int
        :return: generator that yields the entire dataset once
        :rtype: generator
        """

        data_loader = DataLoader(
            dataset=self, shuffle=shuffle, num_workers=n_workers
        )
        for sample in cycle(data_loader):
            sample_batch_dim_removed = {}
            for key, val in sample.items():
                sample_batch_dim_removed[key] = val[0]
            yield sample_batch_dim_removed
