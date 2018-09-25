"""Trainer for training a model on ImageNet"""

from tensorflow.keras import Model


class ImageNetTrainer(object):
    """ImageNet Trainer"""

    def __init__(self, trainer_config):
        """Init

        trainer_config must contain the following keys:
        - str optimizer: optimizer to use when training the network
        - str loss: loss function to use when training the network
        - int batch_size: batch size to use during training
        - int num_epochs: number of epochs to train for

        :param trainer_config: specifies the configuration of the trainer
        :type trainer_config: dict
        """

        self._validate_config(trainer_config)

        self.optimizer = trainer_config['optimizer']
        self.loss = trainer_config['loss']
        self.batch_size = trainer_config['batch_size']
        self.num_epochs = trainer_config['num_epochs']

    @staticmethod
    def _validate_config(trainer_config):
        """Validate that the necessary keys are in the trainer_config

        This raises a KeyError if there are required keys that are missing, and
        otherwise does nothing.

        :param trainer_config: specifies the configuration for the trainer
        :type trainer_config: dict
        """

        required_keys = {'optimizer', 'loss', 'batch_size', 'num_epochs'}
        missing_keys = required_keys - set(trainer_config)

        if missing_keys:
            msg = (
                '{} keys are missing from the trainer_config, but are '
                'required in order to use the ImageNetTrainer.'
            ).format(missing_keys)
            raise KeyError(msg)

    def train(self, train_dataset, network, val_dataset=None):
        """Train the network as specified via the __init__ parameters

        :param train_dataset: dataset that iterates over the training data
         indefinitely
        :type train_dataset: tf.data.Dataset
        :param network: network object to use for training
        :type network: networks.alexnet.AlexNet
        :param val_dataset: optional dataset that iterates over the validation
         data indefinitly
        :type val_dataset: tf.data.Dataset
        """

        inputs, outputs = network.build()
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.optimizer, loss=self.loss
        )

        if val_dataset is not None:
            validation_data = val_dataset.get_infinite_iter()
            validation_steps = len(val_dataset) // self.batch_size
        else:
            validation_data = None
            validation_steps = None

        model.fit(
            x=train_dataset.get_infinite_iter(),
            steps_per_epoch=len(train_dataset) // self.batch_size,
            epochs=self.num_epochs,
            verbose=True,
            validation_data=validation_data,
            validation_steps=validation_steps
        )
