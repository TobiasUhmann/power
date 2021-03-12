import logging
from os import makedirs
from os.path import isdir
from pathlib import Path


class BaseDir:
    name: str
    path: Path

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

    def check(self) -> None:
        """
        Check that directory exists, exit if it does not.
        """

        if not isdir(self.path):
            logging.error(f'{self.name} not found')
            exit()

    def create(self) -> None:
        """
        Create directory if it does not exist already.
        """

        makedirs(self.path, exist_ok=True)
