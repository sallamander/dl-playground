"""Unit tests for training.pytorch.imagenet_trainer"""

from unittest.mock import patch, MagicMock

from ktorch.model import Model

from training.pytorch.imagenet_trainer import ImageNetTrainer


class TestImageNetTrainer(object):
    """Tests for ImageNetTrainer"""

    BATCH_SIZE = 3

    def test_init(self, monkeypatch):
        """Test __init__ method

        This tests that all attributes are set correctly in the __init__.

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        def mock_validate_config(config, required_keys):
            """Mock validate_config to pass"""
            pass
        monkeypatch.setattr(
            'training.pytorch.imagenet_trainer.validate_config',
            mock_validate_config
        )

        trainer_config = {
            'optimizer': 'Adam', 'loss': 'CrossEntropyLoss',
            'batch_size': self.BATCH_SIZE, 'n_epochs': 2
        }
        dirpath_save = MagicMock()
        imagenet_trainer = ImageNetTrainer(
            trainer_config, dirpath_save, device='cpu'
        )

        assert imagenet_trainer.dirpath_save == dirpath_save
        assert imagenet_trainer.optimizer == 'Adam'
        assert imagenet_trainer.loss == 'CrossEntropyLoss'
        assert imagenet_trainer.batch_size == self.BATCH_SIZE
        assert imagenet_trainer.n_epochs == 2
        assert imagenet_trainer.device == 'cpu'

    def test_train(self, monkeypatch):
        """Test train method"""

        alexnet = MagicMock()
        imagenet_dataset = MagicMock()

        imagenet_trainer = MagicMock()
        imagenet_trainer.n_epochs = 2
        imagenet_trainer.optimizer = 'Adam'
        imagenet_trainer.loss = 'CrossEntropyLoss'

        mock_compile = MagicMock()
        monkeypatch.setattr(
            'training.pytorch.imagenet_trainer.Model.compile', mock_compile
        )

        mock_cycle = MagicMock()
        mock_cycle_return = MagicMock()
        mock_cycle.return_value = mock_cycle_return
        monkeypatch.setattr(
            'training.pytorch.imagenet_trainer.cycle', mock_cycle
        )

        imagenet_trainer.train = ImageNetTrainer.train
        with patch.object(Model, 'fit_generator') as fit_fn:
            imagenet_trainer.train(
                self=imagenet_trainer,
                train_dataset=imagenet_dataset, network=alexnet,
                n_steps_per_epoch=1
            )
            assert fit_fn.call_count == 1
            fit_fn.assert_called_with(
                generator=mock_cycle_return, n_steps_per_epoch=1,
                n_epochs=2, validation_data=None, n_validation_steps=None,
                callbacks=None
            )
            assert mock_compile.call_count == 1

            # reset call_count for next assert
            mock_compile.call_count = 0

        with patch.object(Model, 'fit_generator') as fit_fn:
            imagenet_trainer.train(
                self=imagenet_trainer,
                train_dataset=imagenet_dataset, network=alexnet,
                n_steps_per_epoch=1,
                validation_dataset=imagenet_dataset, n_validation_steps=3
            )
            assert fit_fn.call_count == 1
            fit_fn.assert_called_with(
                generator=mock_cycle_return, n_steps_per_epoch=1,
                n_epochs=2, validation_data=mock_cycle_return,
                n_validation_steps=3, callbacks=None
            )
            assert mock_compile.call_count == 1
