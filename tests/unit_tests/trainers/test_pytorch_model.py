"""Unit tests for trainers.pytorch_model"""

from itertools import product
from unittest.mock import MagicMock
import pytest

import torch

from trainers.pytorch_model import Model


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

        model = Model(network)
        assert not model.device

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

    def test_fit_generator__compiled(self):
        """Test fit_generator method when the network is compiled first

        This tests that the correct total number of steps are taken for a given
        `fit_generator` call with a specified `n_steps_per_epoch` and
        `n_epochs`.
        """

        def mock_generator_fn():
            """Mock generator yielding (inputs, targets) indefinitely"""

            inputs = torch.randn((2, 64, 64, 3))
            targets = torch.randn((2, 1))

            while True:
                yield (inputs, targets)

        model = MagicMock()
        model.fit_generator = Model.fit_generator
        model._compiled = True
        model.network = MagicMock()
        model.loss = MagicMock()

        test_cases = [
            {'n_steps_per_epoch': 1, 'n_epochs': 1, 'device': 'cpu'},
            {'n_steps_per_epoch': 2, 'n_epochs': 2},
            {'n_steps_per_epoch': 223, 'n_epochs': 50, 'device': 'cpu'}
        ]

        for test_case in test_cases:
            n_steps_per_epoch = test_case['n_steps_per_epoch']
            n_epochs = test_case['n_epochs']
            device = test_case.get('device')

            model.device = device
            model.fit_generator(
                self=model, generator=mock_generator_fn(),
                n_steps_per_epoch=n_steps_per_epoch, n_epochs=n_epochs
            )

            assert model.loss.call_count == n_steps_per_epoch * n_epochs
            assert model.network.call_count == n_steps_per_epoch * n_epochs

            model.loss.call_count = 0
            model.network.call_count = 0

    def test_fit_generator__not_compiled(self):
        """Test fit_generator method when the network is not compiled first

        This simply tests that a RuntimeError is raised if the network is not
        compiled before `fit_generator` is called.
        """

        model = MagicMock()
        model.fit_generator = Model.fit_generator

        model._compiled = False
        with pytest.raises(RuntimeError):
            model.fit_generator(
                self=model, generator=MagicMock(), n_steps_per_epoch=1,
                n_epochs=1
            )
