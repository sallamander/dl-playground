"""Setup file

This doesn't list any dependencies - it exists solely to allow for pip
installs with the develop flag (e.g. `pip install -e dl_playground`).
"""

from setuptools import setup

setup(
    name='dl_playground',
    version='0.1.0',
    description=(
        'A repository for experimenting and tinkering with deep '
        'learning architectures.'
    ),
    author='Sean Sall'
)
