"""Development environment key-value fetcher

This module exposes a single function, `get`, which returns the value of a
given key for a specified group in the `dev_env.ini` file that sits at the
topmost level of this repository.
"""

import os
from configparser import ConfigParser

FPATH_ABSOLUTE = os.path.abspath(__file__)
FPATH_DEV_ENV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(FPATH_ABSOLUTE))),
    'dev_env.ini'
)


def get(group, key):
    """Retrieve the value for the given `key` in the provided `group`

    :param group: name of the group that `key` is in
    :type group: str
    :param key: name of they key to retrieve the value for
    :type key: str
    :return: holds the value for the `group` and `key` combo
    :rtype: str
    """

    parser = ConfigParser()
    parser.read(FPATH_DEV_ENV)
    return os.path.expanduser(parser[group][key])
