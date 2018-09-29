"""Generic utilities used throughout dl-playground."""


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
