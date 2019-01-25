"""Class for running a specified pytorch training job"""

import pandas as pd
from torch.utils.data import DataLoader

from training.pytorch.dataset_transformer import PyTorchDataSetTransformer
from training.training_job import TrainingJob
from utils.generic_utils import import_object


class PyTorchTrainingJob(TrainingJob):
    """Runs a training job as specified via a config"""

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

        dataset = PyTorchDataSetTransformer(dataset, transformations)
        loading_params = dataset_spec['{}_loading_params'.format(set_name)]
        dataset_gen = DataLoader(dataset, **loading_params)
        n_batches = len(dataset) // loading_params['batch_size']

        return dataset_gen, n_batches
