"""Class for training / evaluating pytorch networks

Reference Implementations:
- https://github.com/keras-team/keras/blob/master/keras/engine/training.py
"""

from tensorflow.python.keras.callbacks import (
    BaseLogger, CallbackList, History, ProgbarLogger
)
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

        self.history = History()
        self.stop_training = False

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

    def _default_callbacks(self):
        """Return default callbacks automatically applied during training

        By default, the following callbacks are automatically applied during
        training:
        - tensorflow.keras.callbacks.BaseLogger
        - tensorflow.keras.callbacks.ProgbarLogger
        - tensorflow.keras.callbacks.History (which is the `Model.history`
          attribute set in `Model.__init__`)

        :return: callbacks automatically applied to every Model
        :rtype: list
        """

        default_callbacks = [
            BaseLogger(), ProgbarLogger(count_mode='steps'), self.history
        ]
        return default_callbacks

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

        default_callbacks = self._default_callbacks()
        callbacks = CallbackList(default_callbacks)

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

        callbacks.set_params({
            'epochs': n_epochs,
            'metrics': ['loss', 'val_loss'],
            'steps': n_steps_per_epoch,
            'verbose': True
        })
        callbacks.set_model(self)

        callbacks.on_train_begin()
        for idx_epoch in range(n_epochs):
            if self.stop_training:
                break

            epoch_logs = {}
            callbacks.on_epoch_begin(idx_epoch)

            for idx_batch in range(n_steps_per_epoch):
                batch_logs = {'batch': idx_batch, 'size': 1}
                callbacks.on_batch_begin(idx_batch, batch_logs)

                inputs, targets = next(generator)
                loss = self.train_on_batch(inputs, targets)

                batch_logs['loss'] = loss
                callbacks.on_batch_end(idx_batch, batch_logs)

                if self.stop_training:
                    break

            if validation_data:
                val_loss = self.evaluate_generator(
                    validation_data, n_validation_steps
                )
                epoch_logs['val_loss'] = val_loss
            callbacks.on_epoch_end(idx_epoch, epoch_logs)
        callbacks.on_train_end()

    def load_weights(self, fpath_weights):
        """Loads all layer weights from the provided `fpath_weights`

        :param fpath_weights: fpath_weights to load the model from
        :type fpath_weights: str
        """

        self.network.load_state_dict(torch.load(fpath_weights))

    def save_weights(self, fpath_weights, overwrite=True):
        """Dumps all layers and weights to the provided `fpath_weights`

        The weights can be loaded into a `Model` with the same topology using
        the `Model.load_weights` method.

        :param fpath_weights: fpath_weights to save the model to
        :type fpath_weights: str
        :param overwrite: overwrite an existing file at `fpath_weights`
         (if present); only True is currently supported
        :type overwrite: bool
        """

        assert overwrite, '`overwrite=False` is not supported!'
        torch.save(self.network.state_dict(), fpath_weights)

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
