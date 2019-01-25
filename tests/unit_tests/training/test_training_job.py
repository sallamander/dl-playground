"""Unit tests for training.training_job"""

import os
from unittest.mock import call, create_autospec, MagicMock
import shutil
import tempfile
import pytest

from training.training_job import TrainingJob


class TestTrainingJob(object):
    """Tests for TrainingJob"""

    def test_init(self, monkeypatch):
        """Test __init__ method

        This test tests two things:
        - All attributes are set correctly in the __init__
        - The `validate_config` function is called from the init with the
          expected arguments

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_validate_config = MagicMock()
        monkeypatch.setattr(
            'training.training_job.validate_config', mock_validate_config
        )

        mock_parse_dirpath_job = MagicMock()
        # use `mkdtemp` to ensure that only the calling user has read / write
        # permissions
        dirpath_job = tempfile.mkdtemp()
        mock_parse_dirpath_job.return_value = dirpath_job
        monkeypatch.setattr(
            'training.training_job.TrainingJob._parse_dirpath_job',
            mock_parse_dirpath_job
        )

        expected_fpath_config = os.path.join(dirpath_job, 'config.yml')
        try:
            assert not os.path.exists(expected_fpath_config)
            mock_config = MagicMock()
            training_job = TrainingJob(mock_config)
            assert os.path.exists(expected_fpath_config)
        finally:
            shutil.rmtree(dirpath_job)

        assert id(training_job.config) == id(mock_config)
        assert (
            training_job.required_config_keys ==
            {'network', 'trainer', 'dataset'}
        )
        mock_validate_config.assert_called_once_with(
            mock_config, training_job.required_config_keys
        )

    def test_instantiate_network(self, monkeypatch):
        """Test _instantiate_network method

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_config = {
            'network': {
                'importpath': 'path.to.network.import',
                'init_params': {
                    'config': {'key1': 'value1', 'key2': 'value2'},
                    'test': 0,
                    'params': (1, 2, 3)
                }
            }
        }

        mock_import_object = MagicMock()
        mock_network_class = MagicMock()
        mock_import_object.return_value = mock_network_class
        monkeypatch.setattr(
            'training.training_job.import_object', mock_import_object
        )

        training_job = MagicMock()
        training_job._instantiate_network = TrainingJob._instantiate_network
        training_job.config = mock_config

        training_job._instantiate_network(self=training_job)
        mock_import_object.assert_called_once_with('path.to.network.import')
        mock_network_class.assert_called_once_with(
            test=0, params=(1, 2, 3),
            config={'key1': 'value1', 'key2': 'value2'}
        )

    def test_instantiate_trainer(self, monkeypatch):
        """Test _instantiate_trainer method

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_config = {
            'trainer': {
                'importpath': 'path.to.trainer.import',
                'init_params': {
                    'param0': 0,
                    'param1': (2, 4, 6),
                    'param2': {'key3': 'value3'}
                }
            }
        }

        mock_import_object = MagicMock()
        mock_trainer_class = MagicMock()
        mock_import_object.return_value = mock_trainer_class
        monkeypatch.setattr(
            'training.training_job.import_object', mock_import_object
        )

        training_job = MagicMock()
        training_job.dirpath_job = 'dirpath_job'
        training_job._instantiate_trainer = TrainingJob._instantiate_trainer
        training_job.config = mock_config

        training_job._instantiate_trainer(self=training_job)
        mock_import_object.assert_called_once_with('path.to.trainer.import')
        mock_trainer_class.assert_called_once_with(
            param0=0, param1=(2, 4, 6), param2={'key3': 'value3'},
            dirpath_save='dirpath_job'
        )

    def test_parse_dirpath_job(self, monkeypatch):
        """Test _parse_dirpath_job

        This tests that the correct `dirpath_job` value is returned when both,
        neither, or either of `dirpath_jobs` and `job_name` are included in the
        `config` passed to the `TrainingJob.__init__`.

        `time.strftime` is patched in order to produce a deterministic
        timestamp for testing purposes. `os.environ['HOME']` is manually set in
        order to point to a temporary directory.

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        fixed_utc_time = '2019-01-25_143601'
        # use `mkdtemp` to get a random filename that is only read / writable
        # by the calling user, but delete it in order to test that it is
        # re-created in the _parse_dirpath_job method
        tempdir1 = tempfile.mkdtemp()
        tempdir2 = tempfile.mkdtemp()
        tempdir3 = tempfile.mkdtemp()
        shutil.rmtree(tempdir1)
        shutil.rmtree(tempdir2)
        shutil.rmtree(tempdir3)
        os.environ['HOME'] = tempdir3

        expected_dirpath_job1 = os.path.join(
            tempdir1, 'test1_{}'.format(fixed_utc_time)
        )
        expected_dirpath_job2 = os.path.join(
            tempdir3, 'training_jobs', '{}_{}'.format('test2', fixed_utc_time),
        )
        expected_dirpath_job3 = os.path.join(tempdir2, fixed_utc_time)
        expected_dirpath_job4 = os.path.join(
            tempdir3, 'training_jobs', fixed_utc_time
        )

        mock_configs = [
            {'dirpath_jobs': tempdir1, 'job_name': 'test1',
             'expected_dirpath_job': expected_dirpath_job1},
            {'job_name': 'test2',
             'expected_dirpath_job': expected_dirpath_job2},
            {'dirpath_jobs': tempdir2,
             'expected_dirpath_job': expected_dirpath_job3},
            {'expected_dirpath_job': expected_dirpath_job4}
        ]

        training_job = MagicMock()
        training_job._parse_dirpath_job = TrainingJob._parse_dirpath_job

        mock_strftime = MagicMock()
        mock_strftime.return_value = fixed_utc_time
        monkeypatch.setattr('time.strftime', mock_strftime)

        for config in mock_configs:
            try:
                if 'dirpath_jobs' in config:
                    assert not os.path.exists(config['dirpath_jobs'])
                expected_dirpath_job = config['expected_dirpath_job']
                assert not os.path.exists(expected_dirpath_job)

                training_job.config = config
                dirpath_job = (
                    training_job._parse_dirpath_job(self=training_job)
                )

                assert expected_dirpath_job == dirpath_job
                assert os.path.exists(dirpath_job)
            finally:
                shutil.rmtree(tempdir1, ignore_errors=True)
                shutil.rmtree(tempdir2, ignore_errors=True)
                shutil.rmtree(tempdir3, ignore_errors=True)

    def test_parse_transformations(self, monkeypatch):
        """Test _parse_transformations method

        This tests two things:
        - That an AssertionError is raised if any of the individual elements in
          the `transformations` argument passed to the _parse_transformations
          method are not of length 1
        - That the returned output is as exepcted

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        training_job = MagicMock()
        training_job._parse_transformations = (
            TrainingJob._parse_transformations
        )
        mock_import_object = MagicMock()
        mock_import_object.return_value = 'import_object_return'
        monkeypatch.setattr(
            'training.training_job.import_object', mock_import_object
        )

        with pytest.raises(AssertionError):
            processed_transformations = training_job._parse_transformations(
                self=training_job,
                transformations=[{'key1_OK': {}, 'key2_NOT_OK': {}}]
            )
        assert mock_import_object.call_count == 0

        transformations = [
            {'transformation1':
             {'sample_keys': {'value': ['image', 'label']}}},
            {'transformation2':
             {'sample_keys': {'value': ['image']},
              'dtype': {'value': 'torch.long', 'import': True}}}
        ]

        processed_transformations = training_job._parse_transformations(
            self=training_job, transformations=transformations
        )
        assert mock_import_object.call_count == 3
        mock_import_object.assert_has_calls([
            call('transformation1'), call('transformation2'),
            call('torch.long')
        ])
        assert processed_transformations == [
            ('import_object_return', {'sample_keys': ['image', 'label']}),
            ('import_object_return',
             {'sample_keys': ['image'], 'dtype': 'import_object_return'})
        ]

    def test_run(self):
        """Test run method"""

        training_job = create_autospec(TrainingJob)
        training_job._instantiate_dataset.return_value = ('mock_dataset', 10)
        training_job.run = TrainingJob.run

        training_job.run(self=training_job)
        training_job._instantiate_network.assert_called_once_with()
        training_job._instantiate_trainer.assert_called_once_with()
        training_job._instantiate_dataset.assert_has_calls([
            call(set_name='train'), call(set_name='validation')
        ])

        trainer = training_job._instantiate_trainer()
        trainer.train.assert_called_once_with(
            network=training_job._instantiate_network(),
            train_dataset='mock_dataset', n_steps_per_epoch=10,
            validation_dataset='mock_dataset', n_validation_steps=10
        )
