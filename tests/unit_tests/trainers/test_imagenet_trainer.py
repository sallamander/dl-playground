"""Unit tests for trainers.imagenet_trainer"""

from unittest.mock import patch, MagicMock

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from trainers.imagenet_trainer import ImageNetTrainer


class TestImageNetTrainer(object):
    """Tests for ImageNetTrainer"""

    BATCH_SIZE = 3
    HEIGHT = 227
    WIDTH = 227
    NUM_CHANNELS = 3

    def get_alexnet(self):
        """Return a mock networks.alexnet.AlexNet object

        :return: alexnet model to use during training
        :rtype: unittest.mock.MagicMock
        """

        def mock_build():
            """Return mock `inputs` and `outputs`"""

            inputs = Input(shape=(self.HEIGHT, self.WIDTH, self.NUM_CHANNELS))
            outputs = inputs
            return inputs, outputs

        alexnet = MagicMock()
        alexnet.build = mock_build
        return alexnet

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
            'trainers.imagenet_trainer.validate_config', mock_validate_config
        )

        trainer_config = {
            'optimizer': 'adam', 'loss': 'categorical_crossentropy',
            'batch_size': self.BATCH_SIZE, 'num_epochs': 2
        }
        imagenet_trainer = ImageNetTrainer(trainer_config)

        assert imagenet_trainer.optimizer == 'adam'
        assert imagenet_trainer.loss == 'categorical_crossentropy'
        assert imagenet_trainer.batch_size == self.BATCH_SIZE
        assert imagenet_trainer.num_epochs == 2

    def test_train(self):
        """Test train method"""

        alexnet = self.get_alexnet()
        imagenet_dataset = MagicMock()
        imagenet_dataset.__len__ = MagicMock()
        imagenet_dataset.get_infinite_iter = MagicMock()

        imagenet_trainer = MagicMock()
        imagenet_trainer.optimizer = 'adam'
        imagenet_trainer.loss = 'categorical_crossentropy'

        imagenet_trainer.train = ImageNetTrainer.train
        with patch.object(Model, 'fit') as fit_fn:
            imagenet_trainer.train(
                self=imagenet_trainer,
                train_dataset=imagenet_dataset, network=alexnet
            )
            assert fit_fn.call_count == 1
        assert imagenet_dataset.get_infinite_iter.call_count == 1
        assert imagenet_dataset.__len__.call_count == 1

        # reassign `get_infinite_iter` and `__len__` to reset their call count
        imagenet_dataset.get_infinite_iter = MagicMock()
        imagenet_dataset.__len__ = MagicMock()
        with patch.object(Model, 'fit') as fit_fn:
            imagenet_trainer.train(
                self=imagenet_trainer, train_dataset=imagenet_dataset,
                network=alexnet, val_dataset=imagenet_dataset
            )
            assert fit_fn.call_count == 1
        assert imagenet_dataset.get_infinite_iter.call_count == 2
        assert imagenet_dataset.__len__.call_count == 2
