"""
Functions for checking the file structure of a Ryn Dataset Directory
"""

import pathlib
from os import makedirs
from os.path import isdir

from dao.ryn.split.split_dir import SplitDir
from dao.ryn.text.text_dir import TextDir


class RynDir:
    name: str
    path: pathlib.Path

    split_dir: SplitDir
    text_dir: TextDir

    def __init__(self, name: str, path: pathlib.Path):
        self.name = name
        self.path = path

        self.split_dir = SplitDir('Split Directory', path.joinpath('split'))
        self.text_dir = TextDir('Text Directory', path.joinpath('text'))

    def check(self) -> None:
        if not isdir(self.path):
            print(f'{self.name} not found')
            exit()

        self.split_dir.check()
        self.text_dir.check()

    def create(self) -> None:
        makedirs(self.path, exist_ok=True)

        self.split_dir.create()
        self.text_dir.create()
