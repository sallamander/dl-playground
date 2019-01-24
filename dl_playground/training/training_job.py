"""Class for running a specified training job"""

from utils.generic_utils import import_object, validate_config


class TrainingJob(object):
    """Runs a training job as specified via a config"""

    required_config_keys = {'network', 'trainer', 'dataset'}

    def __init__(self, config):
        """Init

        The `config` must contain the following keys:
        - dict network: specifies the network class to train as well as how to
          build it; see the `_instantiate_network` method for details
        - dict trainer: specifies the trainer class to train with; see the
          `_instantiate_trainer` method for details
        - dict dataset: specifies the training and validation dataset classes
          to train with, as well as how to load the data from the datasets; see
          the `_instantiate_dataset` method in child classes for details

        :param config: config file specifying a training job to run
        :type config: dict
        """

        validate_config(config, self.required_config_keys)
        self.config = config

    def _instantiate_dataset(self, set_name):
        """Return a dataset object to be used as an iterator during training

        The dataset that is returned should be able to be directly passed into
        the `train` method of whatever trainer class is specified in
        `self.config`, as either the `train_dataset` or `validation_dataset`
        argument.

        :param set_name: set to return the dataset for, one of
         {'train', 'validation'}
        :type set_name: str
        """

        raise NotImplementedError

    def _instantiate_network(self):
        """Return the network object to train

        This relies on the `network` section of `self.config`. This section
        must contain the following keys:
        - str importpath: import path to the network class to use for training
        - dict init_params: parameters to pass directly into the `__init__` of
          the specified network as keyword arguments

        :return: network for training
        :rtype: object
        """

        network_spec = self.config['network']

        network_importpath = network_spec['importpath']
        Network = import_object(network_importpath)
        return Network(**network_spec['init_params'])

    def _instantiate_trainer(self):
        """Return the trainer object that runs training

        This relies on the `trainer` section of `self.config`. This section
        must contain the following keys:
        - str importpath: import path to the trainer class to use
        - dict init_params: parameters to pass directly into the `__init__` of
          the specified trainer as keyword arguments

        :return: trainer to run training
        :rtype: object
        """

        trainer_spec = self.config['trainer']

        trainer_importpath = trainer_spec['importpath']
        Trainer = import_object(trainer_importpath)
        return Trainer(**trainer_spec['init_params'])

    def _parse_transformations(self, transformations):
        """Parse the provided transformations into the expected format

        When passed into the dataset transformers (whose import paths are
        listed below), `transformations` is expected to be a list of two
        element tuples, where each tuple contains a transformation function to
        apply as the first element and function kwargs as the second element.
        When they are parsed from the config (and passed into this function),
        they are a list of dictionaries. This function mostly reformats them to
        the format expected by the following dataset transformer classes:
        - training.pytorch.dataset_transformer.PyTorchDataSetTransformer
        - training.tf.data_loader.TFDataLoader

        :param transformations: holds the transformations to apply to each
         batch of data, where each transformation is specified as a dictionary
         with the key equal to the importpath of the callable transformation
         and the value equal to a dictionary holding keyword arguments for the
         callable
        :type transformations: list[dict]
        :return: parsed transformations reformatted for the dataset transformer
         classes
        :type transformations: list[tuple]
        """

        processed_transformations = []
        for transformation in transformations:
            assert len(transformation) == 1
            transformation_fn_importpath = list(transformation.keys())[0]
            transformation_config = list(transformation.values())[0]

            transformation_fn = import_object(transformation_fn_importpath)
            processed_transformation_config = {}
            for param, arguments in transformation_config.items():
                value = arguments['value']
                if arguments.get('import'):
                    value = import_object(value)
                processed_transformation_config[param] = value
            processed_transformations.append(
                (transformation_fn, processed_transformation_config)
            )

        return processed_transformations

    def run(self):
        """Run training as specified via `self.config`"""

        network = self._instantiate_network()
        trainer = self._instantiate_trainer()
        train_dataset, n_steps_per_epoch = (
            self._instantiate_dataset(set_name='train')
        )
        validation_dataset, n_validation_steps = (
            self._instantiate_dataset(set_name='validation')
        )

        trainer.train(
            network=network,
            train_dataset=train_dataset,
            n_steps_per_epoch=n_steps_per_epoch,
            validation_dataset=validation_dataset,
            n_validation_steps=n_validation_steps
        )
