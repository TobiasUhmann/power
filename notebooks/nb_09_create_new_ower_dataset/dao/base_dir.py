import logging
from os import makedirs
from os.path import isdir
from pathlib import Path

from debug import auto_str


@auto_str
class BaseDir:
    _name: str
    _path: Path

    def __init__(self, name: str, path: Path):
        self._name = name
        self._path = path

    def check(self) -> None:
        """
        Check that directory exists, exit if it does not.
        """

        if not isdir(self._path):
            logging.error(f'{self._name} not found')
            exit()

    def create(self) -> None:
        """
        Create directory if it does not exist already.
        """

        makedirs(self._path, exist_ok=True)
