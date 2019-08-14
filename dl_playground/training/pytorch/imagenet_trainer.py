"""Trainer for training a pytorch model on ImageNet"""

from ktorch.model import Model

from utils.generic_utils import cycle, validate_config


class ImageNetTrainer(object):
    """ImageNet Trainer"""

    required_config_keys = {'batch_size', 'loss', 'n_epochs', 'optimizer'}

    def __init__(self, config, dirpath_save, device=None):
        """Init

        config must contain the following keys:
        - str optimizer: optimizer to use when training the network
        - str loss: loss function to use when training the network
        - int batch_size: batch size to use during training
        - int n_epochs: number of epochs to train for

        :param config: specifies the configuration of the trainer
        :type config: dict
        :param dirpath_save: directory path to save the model to during
         training
        :type dirpath_save: str
        :param device: specifies a GPU to use for training
        :type device: str or torch.device
        """

        validate_config(config, self.required_config_keys)

        self.optimizer = config['optimizer']
        self.loss = config['loss']
        self.batch_size = config['batch_size']
        self.n_epochs = config['n_epochs']
        self.device = device

        self.dirpath_save = dirpath_save

    def train(self, network, train_dataset, n_steps_per_epoch,
              validation_dataset=None, n_validation_steps=None, metrics=None,
              callbacks=None):
        """Train the network as specified via the __init__ parameters

        :param network: network object to use for training
        :type network: networks.alexnet_pytorch.AlexNet
        :param train_dataset: dataset that iterates over the training data
         indefinitely
        :type train_data: torch.utils.data.DataLoader
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param validation_dataset: optional dataset that iterates over the
         validation data indefinitely
        :type validation_dataset: torch.utils.data.DataLoader
        :param n_validation_steps: number of batches to validate on after each
         epoch
        :type n_validation_steps: int
        :param metrics: metrics to be evaluated by the model during training
         and testing
        :type metrics: list[object]
        :param callbacks: callbacks to be used during training
        :type callbacks: list[object]
        """

        model = Model(network, self.device)
        model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=metrics
        )

        if validation_dataset:
            validation_dataset = cycle(validation_dataset)

        model.fit_generator(
            generator=cycle(train_dataset),
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs=self.n_epochs,
            validation_data=validation_dataset,
            n_validation_steps=n_validation_steps,
            callbacks=callbacks
        )
