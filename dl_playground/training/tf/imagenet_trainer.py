"""Trainer for training a tensorflow model on ImageNet"""

from tensorflow.keras import Model

from utils.generic_utils import validate_config


class ImageNetTrainer(object):
    """ImageNet Trainer"""

    required_config_keys = {'batch_size', 'loss', 'n_epochs', 'optimizer'}

    def __init__(self, config, dirpath_save):
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
        """

        validate_config(config, self.required_config_keys)

        self.optimizer = config['optimizer']
        self.loss = config['loss']
        self.batch_size = config['batch_size']
        self.n_epochs = config['n_epochs']

        self.dirpath_save = dirpath_save

    def train(self, network, train_dataset, n_steps_per_epoch,
              validation_dataset=None, n_validation_steps=None, metrics=None,
              callbacks=None):
        """Train the network as specified via the __init__ parameters

        :param network: network object to use for training
        :type network: networks.alexnet_tf.AlexNet
        :param train_dataset: dataset that iterates over the training data
         indefinitely
        :type train_data: tf.data.Dataset
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param validation_dataset: optional dataset that iterates over the
         validation data indefinitely
        :type validation_dataset: tf.data.Dataset
        :param n_validation_steps: number of batches to validate on after each
         epoch
        :type n_validation_steps: int
        :param metrics: metrics to be evaluated by the model during training
         and testing
        :type metrics: list[object]
        :param callbacks: callbacks to be used during training
        :type callbacks: list[object]
        """

        inputs, outputs = network.forward()
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=metrics
        )

        model.fit(
            x=train_dataset,
            steps_per_epoch=n_steps_per_epoch,
            epochs=self.n_epochs,
            verbose=True,
            validation_data=validation_dataset,
            validation_steps=n_validation_steps,
            callbacks=callbacks
        )
