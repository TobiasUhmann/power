from os.path import isfile
from pathlib import Path


class BaseFile:
    _name: str
    _path: Path

    def __init__(self, name: str, path: Path):
        self._name = name
        self._path = path

    def check(self) -> None:
        """
        Check that file exists, exit if it does not.
        """

        if not isfile(self._path):
            print(f'{self._name} not found')
            exit()
