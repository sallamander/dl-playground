"""Generic utilities used throughout dl-playground."""

import importlib


def cycle(iterable):
    """Cycle over `iterable` without caching individual iterable elements

    The `itertools.cycle` function caches the results of iterable elements
    during the first pass, and in subsequent passes uses the cached results.
    For iterables to return non-deterministic output, the caching must not be
    used.

    Reference implementation: https://stackoverflow.com/a/49987606

    :param iterable: iterable object to cycle over; must implement the
     `__iter__` magic method
    :type iterable: Iterable
    """

    if not hasattr(iterable, '__iter__'):
        msg = ('`iterable` must implement the `__iter__` magic method, but '
               'doesn\'t.')
        raise AttributeError(msg)

    while True:
        for element in iterable:
            yield element


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
