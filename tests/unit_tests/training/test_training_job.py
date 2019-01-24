"""Unit tests for training.training_job"""

from unittest.mock import call, MagicMock
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

        mock_config = MagicMock()
        training_job = TrainingJob(mock_config)
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
        training_job._instantiate_trainer = TrainingJob._instantiate_trainer
        training_job.config = mock_config

        training_job._instantiate_trainer(self=training_job)
        mock_import_object.assert_called_once_with('path.to.trainer.import')
        mock_trainer_class.assert_called_once_with(
            param0=0, param1=(2, 4, 6), param2={'key3': 'value3'}
        )

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
