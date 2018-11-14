"""Generic utilities used throughout dl-playground."""

import importlib


def import_object(object_importpath):
    """Return the object specified by `object_importpath`

    :param object_importpath: import path to the object to import
    :type object_importpath: str
    :return: object at `object_importpath`
    :rtype: object
    """

    module_path, object_name = object_importpath.rsplit('.', 1)
    module = importlib.import_module(module_path)

    return getattr(module, object_name)


def validate_config(config, required_keys):
    """Validate that the `config` has the `required_keys`

    :param config: specifies the configuration to validate
    :type config: dict
    :param required_keys: keys that are required to be in `config`
    :type required_keys: list
    :raises KeyError: if there are required keys that are missing
    """

    missing_keys = set(required_keys) - set(config)

    if missing_keys:
        msg = '{} keys are missing, but are required.'.format(missing_keys)
        raise KeyError(msg)
