"""Trainer for training a tensorflow model on ImageNet"""

from tensorflow.keras import Model

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

        :param trainer_config: specifies the configuration of the trainer
        :type trainer_config: dict
        """

        validate_config(trainer_config, self.required_config_keys)

        self.optimizer = trainer_config['optimizer']
        self.loss = trainer_config['loss']
        self.batch_size = trainer_config['batch_size']
        self.n_epochs = trainer_config['n_epochs']

    def train(self, network, train_dataset, n_steps_per_epoch,
              validation_dataset=None, n_validation_steps=None):
        """Train the network as specified via the __init__ parameters

        :param network: network object to use for training
        :type network: networks.alexnet_tf.AlexNet
        :param train_dataset: dataset that iterates over the training data
         indefinitely
        :type train_data: tf.data.Dataset
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param validation_dataset: optional dataset that iterates over the
         validation data indefinitly
        :type validation_dataset: tf.data.Dataset
        :param n_validation_steps: number of batches to validate on after each
         epoch
        :type n_validation_steps: int
        """

        inputs, outputs = network.forward()
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.optimizer, loss=self.loss)

        model.fit(
            x=train_dataset,
            steps_per_epoch=n_steps_per_epoch,
            epochs=self.n_epochs,
            verbose=True,
            validation_data=validation_dataset,
            validation_steps=n_validation_steps
        )
