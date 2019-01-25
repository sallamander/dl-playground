"""Unit tests for training.pytorch.training_job"""

from unittest.mock import MagicMock
import pytest

import pandas as pd

from training.pytorch.training_job import PyTorchTrainingJob


class TestPyTorchTrainingJob(object):
    """Tests for PyTorchTrainingJob"""

    def _get_mock_config(self):
        """Return a mock config to use for a PyTorchTrainingJob

        This is implemented as a callable method (rather than a fixture)
        because we want to alter it in several of the tests, without it
        affecting the other tests.

        :return: mock config that includes a 'dataset' key and the
         configuration necessary to test the _instantiate_dataset method
        :rtype: dict
        """

        return {'dataset': {
            'fpath_df_train': 'fpath/df/train',
            'fpath_df_validation': 'fpath/df/validation',
            'importpath': 'path/to/import',
            'init_params': {'key1': 'value1', 'key2': 'value2'},
            'train_transformations': {'key3': 'value3', 'key4': 'value4'},
            'validation_transformations': {'key5': 'value5', 'key6': 'value6'},
            'train_loading_params': {'batch_size': 2},
            'validation_loading_params': {'batch_size': 4}
        }}

    def _check_instantiate_dataset(self, training_job, set_name, monkeypatch):
        """Check the `_instantiate_dataset` method for the provided `set_name`

        This tests two things:
        - The right functions / classes are used in the method itself
        - The returned output is as expected

        `pd.read_csv` is mocked to prevent tests from relying on a saved
        DataFrame.

        :param training_job: mock training job object to test
         `_instantiate_dataset` against (the `_instantiate_dataset` is not
         mocked)
        :type training_job: unittest.mock.MagicMock
        :param set_name: set to test the method for, one of 'train',
         'validation'
        :type set_name: str
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        training_job._parse_transformations = MagicMock()
        training_job._parse_transformations.return_value = (
            'return_from_parse_transformations'
        )

        mock_read_csv = MagicMock()
        mock_read_csv.return_value = 'return_from_read_csv'
        monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

        MockDataSet = MagicMock()
        MockDataSet.return_value = 'return_from_dataset_init'
        mock_import_object = MagicMock()
        mock_import_object.return_value = MockDataSet
        monkeypatch.setattr(
            'training.pytorch.training_job.import_object', mock_import_object
        )

        mock_transformer = MagicMock()
        mock_transformer_return = MagicMock()
        mock_transformer_return.__len__ = MagicMock()
        mock_transformer_return.__len__.return_value = 10
        mock_transformer.return_value = mock_transformer_return
        monkeypatch.setattr(
            'training.pytorch.training_job.PyTorchDataSetTransformer',
            mock_transformer
        )

        mock_loader = MagicMock()
        mock_loader_return = MagicMock()
        mock_loader.return_value = mock_loader_return
        monkeypatch.setattr(
            'training.pytorch.training_job.DataLoader', mock_loader
        )

        dataset_gen, n_batches = training_job._instantiate_dataset(
            self=training_job, set_name=set_name
        )

        assert dataset_gen == mock_loader_return
        if set_name == 'train':
            assert n_batches == 5
            mock_read_csv.assert_called_once_with('fpath/df/train')
            training_job._parse_transformations.assert_called_once_with(
                {'key3': 'value3', 'key4': 'value4'}
            )
            mock_transformer.assert_called_once_with(
                'return_from_dataset_init', 'return_from_parse_transformations'
            )
            mock_loader.assert_called_once_with(
                mock_transformer_return, batch_size=2
            )
        else:
            assert n_batches == 2
            mock_read_csv.assert_called_once_with('fpath/df/validation')
            training_job._parse_transformations.assert_called_once_with(
                {'key5': 'value5', 'key6': 'value6'}
            )
            mock_transformer.assert_called_once_with(
                'return_from_dataset_init', 'return_from_parse_transformations'
            )
            mock_loader.assert_called_once_with(
                mock_transformer_return, batch_size=4
            )

        mock_import_object.assert_called_once_with('path/to/import')
        MockDataSet.assert_called_once_with(
            df_obs='return_from_read_csv', key1='value1', key2='value2'
        )

    def test_instantiate_dataset__errors(self):
        """Test _instantiate_dataset method when errors are expected

        Errors are expected in two cases:
        - When `set_name` is not one of 'train' or 'validation'
          (AssertionError)
        - When `set_name==train` and there is no `fpath_df_train` in the
          `TFTrainingJob.config['dataset']` (RunTimeError)
        """

        training_job = MagicMock()
        training_job.config = self._get_mock_config()
        training_job._instantiate_dataset = (
            PyTorchTrainingJob._instantiate_dataset
        )

        for bad_set_name in ['test', 'val', 't']:
            with pytest.raises(AssertionError):
                training_job._instantiate_dataset(
                    self=training_job, set_name=bad_set_name
                )

        del training_job.config['dataset']['fpath_df_train']
        with pytest.raises(RuntimeError):
            training_job._instantiate_dataset(
                self=training_job, set_name='train'
            )

    def test_instantiate_dataset__no_errors(self, monkeypatch):
        """Test _instantiate_dataset method when no errors are expected

        This tests two things via the _check_instantiate_dataset method:
        - The right functions / classes are used in the method itself
        - The returned output is as expected

        It checks these for when `set_name` is 'train' and 'validation'.
        Additionally, this tests that when there is no `fpath_df_validation`
        specified, there is no dataset returned.

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        training_job = MagicMock()
        training_job.config = self._get_mock_config()
        training_job._instantiate_dataset = (
            PyTorchTrainingJob._instantiate_dataset
        )

        for set_name in ('train', 'validation'):
            self._check_instantiate_dataset(
                training_job, set_name, monkeypatch
            )

        del training_job.config['dataset']['fpath_df_validation']
        dataset_gen, n_batches = training_job._instantiate_dataset(
            self=training_job, set_name='validation'
        )
        assert dataset_gen is None
        assert n_batches is None
