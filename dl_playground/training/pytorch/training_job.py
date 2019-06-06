"""Class for running a specified pytorch training job"""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.augmented_dataset import AugmentedDataset
from training.training_job import TrainingJob
from utils.generic_utils import import_object


def format_batch(batch, input_keys, target_keys):
    """Format the batch from a list of dictionaries to a tuple of tensors

    :param batch: batch of inputs and targets
	:type batch: list[dict]
    :param input_keys: names of the keys in the elements of `batch` that are
     inputs to a model
	:type input_keys: list[str]
    :param target_keys: names of the keys in `batch` that are targets for a
     model
	:type target_keys: list[str]
    :return: 2-element tuple holding the inputs and targets
    :rtype: tuple(torch.Tensor)
    """

    assert len(input_keys) == 1, 'More than one input_key is not supported.'
    assert len(target_keys ) == 1, 'More than one target_key is not supported.'

    inputs = []
    targets = []
    for element in batch:
        inputs.append(element[input_keys[0]])
        targets.append(element[target_keys[0]])

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return inputs, targets


class PyTorchTrainingJob(TrainingJob):
    """Runs a training job as specified via a config"""

    def __init__(self, config):
        """Init

        The `config` key is passed through to the `super().__init__()`; see
        that method for details.

        This method is over-ridden to update the `trainer` key of the config
        with the device to use, if specified in the `config` itself.
        """

        super().__init__(config)

        if self.gpu_id is not None:
            if not torch.cuda.is_available():
                msg = (
                    'gpu_id {} was specified, but '
                    '`torch.cuda.is_available=False.'
                ).format(self.gpu_id)
                raise RuntimeError(msg)
            device = torch.device('cuda:0')
            self.config['trainer']['init_params']['device'] = device

    def _instantiate_dataset(self, set_name):
        """Return a dataset object to be used as an iterator during training

        The dataset that is returned should be able to be directly passed into
        the `train` method of whatever trainer class is specified in
        `self.config`, as either the `train_dataset` or `validation_dataset`
        argument.

        :param set_name: set to return the dataset for, one of
         {'train', 'validation'}
        :type set_name: str
        :return: two element tuple holding an iterable over the dataset for
         `set_name`, as well as the number of batches in a single pass over the
         dataset
        :rtype: tuple
        """

        assert set_name in {'train', 'validation'}
        dataset_spec = self.config['dataset']

        fpath_df_obs_key = 'fpath_df_{}'.format(set_name)
        if fpath_df_obs_key not in dataset_spec:
            if set_name == 'train':
                raise RuntimeError
            return None, None
        fpath_df_obs = dataset_spec[fpath_df_obs_key]
        df_obs = pd.read_csv(fpath_df_obs)

        dataset_importpath = dataset_spec['importpath']
        DataSet = import_object(dataset_importpath)

        dataset = DataSet(df_obs=df_obs, **dataset_spec['init_params'])
        transformations_key = '{}_transformations'.format(set_name)
        transformations = dataset_spec[transformations_key]
        transformations = self._parse_transformations(transformations)

        dataset = AugmentedDataset(dataset, transformations)
        loading_params = dataset_spec['{}_loading_params'.format(set_name)]

        collate_fn = (
            lambda batch: format_batch(
                batch, dataset.input_keys, dataset.target_keys
            )
        )
        dataset_gen = DataLoader(dataset, collate_fn=collate_fn, **loading_params)
        n_batches = len(dataset) // loading_params['batch_size']

        return dataset_gen, n_batches
