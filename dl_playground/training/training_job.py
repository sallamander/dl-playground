"""Class for running a specified training job"""

import inspect
import os
import time

import pandas as pd
import yaml

from utils.generic_utils import import_object, validate_config

# TODO: Write a unit test for the new `resume` method


class TrainingJob(object):
    """Runs a training job as specified via a config"""

    required_config_keys = {'network', 'trainer', 'dataset'}

    def __init__(self):
        """Init"""

        # these are set when the `resume` or `run` method is called
        self.config = None
        self.dirpath_job = None
        self.gpu_id = None

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
        trainer = Trainer(
            **trainer_spec['init_params'], dirpath_save=self.dirpath_job
        )
        return trainer

    def _parse_callbacks(self):
        """Return the callback objects used during training

        This relies on the `trainer::callbacks` section of `self.config`. This
        section contains a list of callbacks, where each callback is specified
        as a dictionary or a string. If a dictionary, they key is expected to
        be the importpath to the callback, and the value a dictionary of
        __init__ params; if a str only, it is expected to be the importpath to
        the callback.

        :return: callback objects used during training
        :rtype: list[object]
        """

        callback_specs = self.config['trainer'].get('callbacks', [])

        callbacks = []
        for callback_spec in callback_specs:
            if isinstance(callback_spec, dict):
                assert len(callback_spec) == 1
                callback_importpath = list(callback_spec.keys())[0]
                callback_params = list(callback_spec.values())[0]
            else:
                callback_importpath = callback_spec
                callback_params = {}

            if 'CSVLogger' in callback_importpath:
                filename = callback_params.get('filename', 'history.csv')
                callback_params['filename'] = os.path.join(
                    self.dirpath_job, filename
                )
            elif 'TensorBoard' in callback_importpath:
                callback_params['log_dir'] = os.path.join(
                    self.dirpath_job, 'tensorboard'
                )
            elif 'ModelCheckpoint' in callback_importpath:
                default_filename = (
                    'weights.pt' if 'PyTorch' in str(type(self)) else
                    'weights.h5'
                )
                filename = callback_params.get('filename', default_filename)
                callback_params['filepath'] = os.path.join(
                    self.dirpath_job, filename
                )
                callback_params['save_weights_only'] = True

            Callback = import_object(callback_importpath)
            callback = Callback(**callback_params)
            callbacks.append(callback)

        return callbacks

    def _parse_dirpath_job(self):
        """Return the directory path used to save the job

        Anything saved during training (model weights, history CSVs, config
        YMLs, etc.) will be saved in the directory path returned by this
        method.

        :return: directory path to save the job outputs in
        :rtype: str
        """

        default_dirpath_jobs = os.path.join(
            os.environ['HOME'], 'training_jobs'
        )
        dirpath_jobs = self.config.get('dirpath_jobs', default_dirpath_jobs)

        job_timestamp = time.strftime('%Y-%m-%d_%H%M%S', time.gmtime())
        dirname_job = self.config.get('job_name')
        if dirname_job:
            dirname_job = '{}_{}'.format(dirname_job, job_timestamp)
        else:
            dirname_job = job_timestamp

        dirpath_job = os.path.join(dirpath_jobs, dirname_job)
        os.makedirs(dirpath_job, exist_ok=True)
        return dirpath_job

    def _parse_metrics(self):
        """Return the metric objects used during training

        This relies on the `trainer::metrics` section of `self.config`. This
        section contains a list of metrics, where each metric is specified as a
        dictionary or a string. If a dictionary, they key is expected to be the
        importpath to the metric, and the value a dictionary of __init__
        params; if a str only, it is expected to be the importpath to the
        metric.

        :return: metric objects used during training
        :rtype: list[object]
        """

        metric_specs = self.config['trainer'].get('metrics', [])

        metrics = []
        for metric_spec in metric_specs:
            if isinstance(metric_spec, dict):
                assert len(metric_spec) == 1
                metric_importpath = list(metric_spec.keys())[0]
                metric_params = list(metric_spec.values())[0]
            else:
                metric_importpath = metric_spec
                metric_params = {}

            metric_fn = import_object(metric_importpath)
            if inspect.isclass(metric_fn):
                metric_fn = metric_fn(**metric_params)
            metrics.append(metric_fn)

        return metrics

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

    def _run(self, initial_epoch):
        """Run training from `initial_epoch` as specified via `self.config`

        :param initial_epoch: epoch at which to start training
        :type initial_epoch: int
        """

        self.gpu_id = self.config.get('gpu_id', None)
        if self.gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        network = self._instantiate_network()
        trainer = self._instantiate_trainer()
        train_dataset, n_steps_per_epoch = (
            self._instantiate_dataset(set_name='train')
        )
        validation_dataset, n_validation_steps = (
            self._instantiate_dataset(set_name='validation')
        )

        n_steps_per_epoch = 10
        n_validation_steps = 10
        callbacks = self._parse_callbacks()
        metrics = self._parse_metrics()
        trainer.train(
            network=network,
            train_dataset=train_dataset,
            n_steps_per_epoch=n_steps_per_epoch,
            validation_dataset=validation_dataset,
            n_validation_steps=n_validation_steps,
            metrics=metrics,
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )

    def resume(self, dirpath_job):
        """Resume training from the provided job directory

        Training will resume from the most recent epoch completed. If no full
        epochs of training were completed, this method will start with initial
        epoch 0. Further, if training was interrupted before a config file was
        saved, such that there is no config in `dirpath_job`, this method will
        raise a RuntimeError.

        :param dirpath_job: directory path of a previously started training job
        :type dirpath_job: str
        :raises: RuntimeError, when there is no 'config.yml' in `dirpath_job`
        """

        self.dirpath_job = dirpath_job
        fpath_config = os.path.join(dirpath_job, 'config.yml')
        try:
            with open(fpath_config) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            msg = (
                'When using the `resume` method the `dirpath_job` must '
                'contain a config.yml, and {} does not contain one.'
            ).format(self.dirpath_job)
            raise FileNotFoundError(msg)

        validate_config(self.config, self.required_config_keys)

        fpath_history = os.path.join(self.dirpath_job, 'history.csv')
        try:
            df_history = pd.read_csv(fpath_history)
            initial_epoch = len(df_history)
        except FileNotFoundError:
            initial_epoch = 0

        self._run(initial_epoch)

    def run(self, config):
        """Run training as specified via config

        The `config` must contain the following keys:
        - dict network: specifies the network class to train as well as how to
          build it; see the `_instantiate_network` method for details
        - dict trainer: specifies the trainer class to train with; see the
          `_instantiate_trainer` method for details
        - dict dataset: specifies the training and validation dataset classes
          to train with, as well as how to load the data from the datasets; see
          the `_instantiate_dataset` method in child classes for details

        It can contain the following additional keys:
        - str 'job_name': optional name given to the job; the timestamp of when
          the job started will be appended to the job_name to uniquely identify
          the directory name the job will be saved to
        - str 'dirpath_jobs': optional directory path to save job directory in,
          resulting in the job being saved to 'dirpath_jobs/dirname_job';
          defaults to `os.environ['HOME']/training_jobs`
        - int gpu_id: GPU to run the job on; defaults to None, which means the
          job runs on the CPU

        See the `_parse_dirpath_job` method for details on where the results of
        the training job will be stored.

        :param config: config file specifying a training job to run
        :type config: dict
        """

        validate_config(config, self.required_config_keys)
        self.config = config
        self.dirpath_job = self._parse_dirpath_job()

        fpath_config = os.path.join(self.dirpath_job, 'config.yml')
        with open(fpath_config, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self._run(initial_epoch=0)
