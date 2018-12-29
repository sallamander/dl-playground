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

    def _assert_compiled(self):
        """Raise a value error if the model is not compiled

        This is a convenience wrapper to avoid duplicating these lines in
        multiple methods.

        :raises: RuntimeError if `self._compiled` is not True
        """

        if not self._compiled:
            msg = ('Model must be compiled before training; please call '
                   'the `compile` method before training.')
            raise RuntimeError(msg)

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

    def evaluate_generator(self, generator, n_steps):
        """Evaluate the network on batches of data generated from `generator`

        :param generator: a generator yielding batches indefinitely, where each
         batch is a tuple of (inputs, targets)
        :type generator: generator
        :param n_steps: number of batches to evaluate on
        :type n_steps: int
        :return: scalar test loss
        :rtype: float
        """

        self._assert_compiled()

        if self.device:
            self.network.to(self.device)

        total_loss = 0
        n_obs = 0
        for _ in range(n_steps):
            inputs, targets = next(generator)
            n_obs += inputs.shape[0]

            loss = self.test_on_batch(inputs, targets)
            total_loss += loss

        return total_loss / n_obs

    def fit_generator(self, generator, n_steps_per_epoch, n_epochs=1,
                      validation_data=None, n_validation_steps=None):
        """Train the network on batches of data generated from `generator`

        :param generator: a generator yielding batches indefinitely, where each
         batch is a tuple of (inputs, targets)
        :type generator: generator
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param n_epochs: number of epochs to train the model
        :type n_epochs: int
        :param validation_data: generator yielding batches to evaluate the loss
         on at the end of each epoch, where each batch is a tuple of (inputs,
         targets)
        :type validation_data: generator
        :param n_validation_steps: number of batches to evaluate on from
         `validation_data`
        :raises RuntimeError: if only one of `validation_data` and
         `n_validation_steps` are passed in
        """

        self._assert_compiled()

        invalid_inputs = (
            (validation_data is not None and n_validation_steps is None) or
            (n_validation_steps is not None and validation_data is None)
        )
        if invalid_inputs:
            msg = (
                '`validation_data` and `n_validation_steps` must both be '
                'passed, or neither.'
            )
            raise RuntimeError(msg)

        if self.device:
            self.network.to(self.device)

        for _ in range(n_epochs):
            for _ in range(n_steps_per_epoch):
                inputs, targets = next(generator)
                _ = self.train_on_batch(inputs, targets)

            if validation_data:
                _ = self.evaluate_generator(
                    validation_data, n_validation_steps
                )

    def test_on_batch(self, inputs, targets):
        """Evaluate the model on a single batch of samples

        :param inputs: inputs to predict on
        :type inputs: torch.Tensor
        :param targets: targets to compare model predictions to
        :type targets: torch.Tensor
        :return: scalar test loss
        :rtype: float
        """

        self._assert_compiled()

        self.network.train(mode=False)
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        outputs = self.network(inputs)
        loss = self.loss(outputs, targets)

        return loss.tolist()

    def train_on_batch(self, inputs, targets):
        """Run a single forward / backward pass on a single batch of data

        :param inputs: inputs to use in the forward / backward pass
        :type inputs: torch.Tensor
        :param targets: targets to use in the forward / backward pass
        :type targets: torch.Tensor
        :return: scalar training loss
        :rtype: float
        """

        self._assert_compiled()

        self.network.train(mode=True)
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        outputs = self.network(inputs)
        loss = self.loss(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.tolist()
