import logging
from os.path import isfile
from pathlib import Path


class BaseFile:
    name: str
    path: Path

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

    def check(self) -> None:
        """
        Check that file exists, exit if it does not.
        """

        if not isfile(self.path):
            logging.error(f'{self.name} not found')
            exit()
