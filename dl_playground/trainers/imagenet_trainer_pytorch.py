"""Trainer for training a pytorch model on ImageNet"""

from trainers.pytorch_model import Model

from utils.generic_utils import validate_config


class ImageNetTrainer(object):
    """ImageNet Trainer"""

    required_config_keys = {'batch_size', 'loss', 'n_epochs', 'optimizer'}

    def __init__(self, trainer_config):
        """Init

        trainer_config must contain the following keys:
        - str optimizer: optimizer to use when training the network
        - str loss: loss function to use when training the network
        - int batch_size: batch size to use during training
        - int n_epochs: number of epochs to train for

        trainer_config can addtionally optionally contain the following keys:
        - str device: device to train the model on, e.g. 'cuda:0'; defaults to
          'cpu'

        :param trainer_config: specifies the configuration of the trainer
        :type trainer_config: dict
        """

        validate_config(trainer_config, self.required_config_keys)

        self.optimizer = trainer_config['optimizer']
        self.loss = trainer_config['loss']
        self.batch_size = trainer_config['batch_size']
        self.n_epochs = trainer_config['n_epochs']
        self.device = trainer_config.get('device')

    def train(self, network, train_dataset, n_steps_per_epoch):
        """Train the network as specified via the __init__ parameters

        :param network: network object to use for training
        :type network: networks.alexnet_pytorch.AlexNet
        :param train_dataset: dataset that iterates over the training data
         indefinitely
        :type train_data: tf.data.Dataset
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        """

        model = Model(network, self.device)
        model.compile(optimizer=self.optimizer, loss=self.loss)

        model.fit_generator(
            generator=train_dataset,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs=self.n_epochs,
        )
