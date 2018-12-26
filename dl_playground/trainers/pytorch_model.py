"""Class for training / evaluating pytorch networks

Reference Implementations:
- https://github.com/keras-team/keras/blob/master/keras/engine/training.py
"""

import torch


class Model(object):
    """Model for training / evaluating pytorch networks

    Reference Implementation:
    - https://github.com/keras-team/keras/blob/master/keras/engine/training.py
    """

    def __init__(self, network, device=None):
        """Init

        :param network: pytorch network to train or evaluate
        :type network: torch.nn.Module
        :param device: device to train the network on, e.g. 'cuda:0'
        :type device: str
        """

        self.network = network
        self.device = device

        self._compiled = False
        # these are set in the `compile` method
        self.optimizer = None
        self.loss = None

    def compile(self, optimizer, loss):
        """Setup the model for training

        This sets `self.optimizer` and `self.loss` in place.

        :param optimizer: class name of the optimizer to use when training, one
         of those from `torch.optim` (e.g. `Adam`)
        :type optimizer: str
        :param loss: class name of the loss to use when training, one of those
         from `torch.nn` (e.g. `CrossEntropyLoss`)
        :type loss: str
        :raises AttributeError: if an invalid optimizer or loss function is
         specified
        """

        try:
            Optimizer = getattr(torch.optim, optimizer)
        except AttributeError:
            msg = (
                '`optimizer` must be a `str` representing an optimizer from '
                'the `torch.optim` package, and {} is not a valid one.'
            )
            raise AttributeError(msg.format(optimizer))
        self.optimizer = Optimizer(self.network.parameters())

        try:
            Loss = getattr(torch.nn, loss)
        except AttributeError:
            msg = (
                '`loss` must be a `str` representing a loss from '
                'the `torch.nn` package, and {} is not a valid one.'
            )
            raise AttributeError(msg.format(loss))
        self.loss = Loss()

        self._compiled = True

    def fit_generator(self, generator, n_steps_per_epoch, n_epochs=1):
        """Train the network on data generated batch-by-batch from `generator`

        :param generator: a generator yielding batches indefinitely, where each
         batch is a tuple of (inputs, targets)
        :type generator: generator
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param n_epochs: number of epochs to train the model
        :type n_epochs: int
        """

        if not self._compiled:
            msg = ('Model must be compiled before training; please call '
                   'the `compile` method before training.')
            raise RuntimeError(msg)

        if self.device:
            self.network.to(self.device)

        for _ in range(n_epochs):
            for _ in range(n_steps_per_epoch):
                inputs, targets = next(generator)
                if self.device:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                outputs = self.network(inputs)
                loss = self.loss(outputs, targets.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
