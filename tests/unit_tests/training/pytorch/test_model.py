"""Unit tests for training.pytorch.model"""

from itertools import product
from unittest.mock import create_autospec, patch, MagicMock
import pytest

import numpy as np
from tensorflow.python.keras.callbacks import (
    BaseLogger, CallbackList, History, ProgbarLogger
)
import torch

from training.pytorch.model import Model


class TestModel(object):
    """Tests for Model"""

    def test_init(self):
        """Test __init__ method

        This tests that all attributes are set correctly in the __init__.
        """

        network = MagicMock()
        device = MagicMock()

        model = Model(network, device)
        assert id(network) == id(model.network)
        assert id(device) == id(model.device)
        assert not model._compiled
        assert not model.optimizer
        assert not model.loss
        assert model.history
        assert isinstance(model.history, History)

        model = Model(network)
        assert not model.device

    def test_assert_compiled(self):
        """Test _assert_compiled method"""

        model = MagicMock()
        model._assert_compiled = Model._assert_compiled

        model._compiled = False
        with pytest.raises(RuntimeError):
            model._assert_compiled(self=model)

        model._compiled = True
        model._assert_compiled(self=model)

    def test_default_callbacks(self):
        """Test _default_callbacks method"""

        model = MagicMock()
        model.history = MagicMock()
        model._default_callbacks = Model._default_callbacks

        callbacks = model._default_callbacks(self=model)
        assert isinstance(callbacks[0], BaseLogger)
        assert isinstance(callbacks[1], ProgbarLogger)
        assert id(callbacks[2]) == id(model.history)

    def test_compile(self):
        """Test compile method

        This tests several things:
        - An `AttributeError` is raised if an invalid optimizer or loss
          function is passed in
        - `Model.loss` and `Model.optimizer` are set correctly when a valid
          optimizer and loss are passed in
        - `Model._compiled` is True after `compile` is called
        """

        model = MagicMock()
        model.optimizer = None
        model.loss = None
        model._compiled = False
        model.compile = Model.compile

        network = MagicMock()
        parameters_fn = MagicMock()
        parameters_fn.return_value = [
            torch.nn.Parameter(torch.randn((64, 64)))
        ]
        network.parameters = parameters_fn
        model.network = network

        valid_optimizers = ['Adam', 'RMSprop']
        valid_losses = ['BCELoss', 'CrossEntropyLoss', 'L1Loss']

        for optimizer, loss in product(valid_optimizers, valid_losses):
            assert not model.optimizer
            assert not model.loss
            assert not model._compiled

            model.compile(self=model, optimizer=optimizer, loss=loss)

            assert model.optimizer
            assert model.loss
            assert model._compiled

            # reset for next iteration
            model.optimizer = None
            model.loss = None
            model._compiled = False

        with pytest.raises(AttributeError):
            model.compile(self=model, optimizer='BadOptimizer', loss='BCELoss')

        with pytest.raises(AttributeError):
            model.compile(self=model, optimizer='Adam', loss='BadLoss')

    def test_evaluate_generator(self):
        """Test evaluate_generator"""

        model = MagicMock()
        model.network = MagicMock()
        model.test_on_batch = MagicMock()
        model.test_on_batch.return_value = 0.025
        model.device = MagicMock()
        model.evaluate_generator = Model.evaluate_generator

        def generator():
            """Mock generator function"""

            n_obs = 1
            while True:
                inputs = torch.randn((n_obs, 64, 64, 3))
                targets = torch.randn((n_obs, 1))
                n_obs += 1

                yield (inputs, targets)

        test_cases = [
            {'n_steps': 10, 'device': 'cpu', 'expected_loss': 0.00454},
            {'n_steps': 100, 'expected_loss': 0.000505}
        ]

        for test_case in test_cases:
            n_steps = test_case['n_steps']
            device = test_case.get('device')

            model.device = device
            loss = model.evaluate_generator(
                self=model, generator=generator(), n_steps=n_steps
            )

            assert np.allclose(loss, test_case['expected_loss'], atol=1e-4)
            assert model._assert_compiled.call_count == 1

            # re-assign before the next iteration of the loop
            model._assert_compiled.call_count = 0

    def test_fit_generator(self, monkeypatch):
        """Test fit_generator method

        This tests that the correct total number of steps are taken for a given
        `fit_generator` call with a specified `n_steps_per_epoch` and
        `n_epochs`.
        """

        model = MagicMock()
        model.network = MagicMock()
        model.train_on_batch = MagicMock()
        model.train_on_batch.return_value = 4
        model.device = MagicMock()
        model.fit_generator = Model.fit_generator
        model.evaluate_generator = MagicMock()
        model.evaluate_generator.return_value = 2
        model._default_callbacks = MagicMock()
        model._default_callbacks.return_value = [1, 2, 3]

        generator = MagicMock()
        inputs = torch.randn((2, 64, 64, 3))
        targets = torch.randn((2, 1))
        generator.__next__ = MagicMock()
        generator.__next__.return_value = (inputs, targets)

        test_cases = [
            {'n_steps_per_epoch': 1, 'n_epochs': 1, 'device': 'cpu'},
            {'n_steps_per_epoch': 2, 'n_epochs': 2,
             'validation_data': generator, 'n_validation_steps': 10},
            {'n_steps_per_epoch': 223, 'n_epochs': 50, 'device': 'cpu'}
        ]

        for test_case in test_cases:
            n_steps_per_epoch = test_case['n_steps_per_epoch']
            n_epochs = test_case['n_epochs']
            device = test_case.get('device')
            validation_data = test_case.get('validation_data')
            n_validation_steps = test_case.get('n_validation_steps')

            mock_callback_list = MagicMock()
            mock_callbacks = create_autospec(CallbackList)
            mock_callback_list.return_value = mock_callbacks
            monkeypatch.setattr(
                'training.pytorch.model.CallbackList', mock_callback_list
            )

            model.device = device
            model.fit_generator(
                self=model, generator=generator,
                n_steps_per_epoch=n_steps_per_epoch, n_epochs=n_epochs,
                validation_data=validation_data,
                n_validation_steps=n_validation_steps
            )

            assert model._assert_compiled.call_count == 1
            n_batches = n_steps_per_epoch * n_epochs
            assert model.train_on_batch.call_count == n_batches
            model.train_on_batch.assert_called_with(inputs, targets)
            assert generator.__next__.call_count == n_batches

            mock_callback_list.assert_called_with([1, 2, 3])
            mock_callbacks.set_params.assert_called_with(
                {'epochs': n_epochs, 'metrics': ['loss', 'val_loss'],
                 'steps': n_steps_per_epoch, 'verbose': True}
            )
            assert mock_callbacks.on_train_begin.call_count == 1
            assert mock_callbacks.on_epoch_begin.call_count == n_epochs
            assert mock_callbacks.on_batch_begin.call_count == n_batches
            assert mock_callbacks.on_batch_end.call_count == n_batches
            assert mock_callbacks.on_train_end.call_count == 1
            mock_callbacks.on_batch_end.assert_any_call(
                0, {'batch': 0, 'size': 1, 'loss': 4}
            )

            epoch_logs = {}
            if validation_data is not None:
                model.evaluate_generator.assert_called_with(
                    validation_data, n_validation_steps
                )
                epoch_logs['val_loss'] = 2
            mock_callbacks.on_epoch_end.assert_any_call(0, epoch_logs)

            # reset the call counts for the next iteration
            model._assert_compiled.call_count = 0
            model.train_on_batch.call_count = 0
            model.evaluate_generator.call_count = 0
            generator.__next__.call_count = 0

    def test_fit_generator__bad_input(self):
        """Test fit_generator method with bad inputs

        `fit_generator` throws a RuntimeError if only one of `validation_data`
        and `n_validation_steps` are passed in.
        """

        model = MagicMock()
        model._assert_compiled = MagicMock()
        model.fit_generator = Model.fit_generator

        with pytest.raises(RuntimeError):
            model.fit_generator(
                self=model, generator='sentinel1',
                n_steps_per_epoch='sentinel2', n_epochs='sentinel3',
                validation_data='sentinel4'
            )

        with pytest.raises(RuntimeError):
            model.fit_generator(
                self=model, generator='sentinel1',
                n_steps_per_epoch='sentinel2', n_epochs='sentinel3',
                n_validation_steps='sentinel4'
            )

    def test_test_on_batch(self):
        """Test test_on_batch method"""

        model = create_autospec(Model)
        model.device = 'cpu'
        model.test_on_batch = Model.test_on_batch
        model.network = MagicMock()
        model.network.train = MagicMock()

        inputs = torch.randn((2, 3), requires_grad=True)
        targets = torch.randint(size=(2,), high=2, dtype=torch.int64)
        outputs = torch.nn.Sigmoid()(inputs)

        model.loss = MagicMock()
        loss_value = torch.nn.CrossEntropyLoss()(outputs, targets)
        model.loss.return_value = loss_value

        loss = model.test_on_batch(
            self=model, inputs=inputs, targets=targets
        )

        assert loss == loss_value.tolist()
        assert model.network.call_count == 1
        assert model._assert_compiled.call_count == 1
        model.network.train.assert_called_with(mode=False)
        assert model.loss.call_count == 1

    def test_train_on_batch(self):
        """Test train_on_batch method"""

        model = create_autospec(Model)
        model.device = 'cpu'
        model.train_on_batch = Model.train_on_batch
        model.network = MagicMock()
        model.network.train = MagicMock()

        inputs = torch.randn((2, 3), requires_grad=True)
        targets = torch.randint(size=(2,), high=2, dtype=torch.int64)
        outputs = torch.nn.Sigmoid()(inputs)

        model.loss = MagicMock()
        loss_value = torch.nn.CrossEntropyLoss()(outputs, targets)
        model.loss.return_value = loss_value
        model.optimizer = create_autospec(torch.optim.Adam)

        with patch.object(loss_value, 'backward') as patched_backward:
            loss = model.train_on_batch(
                self=model, inputs=inputs, targets=targets
            )

        assert loss == loss_value.tolist()
        assert model.network.call_count == 1
        model.network.train.assert_called_with(mode=True)
        assert model.loss.call_count == 1
        assert model.optimizer.zero_grad.call_count == 1
        assert model.optimizer.step.call_count == 1
        assert patched_backward.call_count == 1
