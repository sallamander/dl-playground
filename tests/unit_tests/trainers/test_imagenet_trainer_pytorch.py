"""Unit tests for trainers.imagenet_trainer_pytorch"""

from unittest.mock import patch, MagicMock

from trainers.pytorch_model import Model
from trainers.imagenet_trainer_pytorch import ImageNetTrainer


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
            'trainers.imagenet_trainer_pytorch.validate_config',
            mock_validate_config
        )

        trainer_config = {
            'optimizer': 'Adam', 'loss': 'CrossEntropyLoss',
            'batch_size': self.BATCH_SIZE, 'n_epochs': 2, 'device': 'cpu'
        }
        imagenet_trainer = ImageNetTrainer(trainer_config)

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
            'trainers.imagenet_trainer_pytorch.Model.compile', mock_compile
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
                generator=imagenet_dataset, n_steps_per_epoch=1,
                n_epochs=2
            )
            assert mock_compile.call_count == 1
