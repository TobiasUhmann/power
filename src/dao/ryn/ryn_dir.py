"""
Functions for checking the file structure of a Ryn Dataset Directory
"""

import os
import pathlib

from dao.ryn.split.split_dir import SplitDir
from dao.ryn.text.text_dir import TextDir


class RynDir:
    path: pathlib.Path

    split_dir: SplitDir
    text_dir: TextDir

    def __init__(self, path: pathlib.Path):
        self.path = path

        self.split_dir = SplitDir(path.joinpath('split'))
        self.text_dir = TextDir(path.joinpath('text'))

    def check(self) -> None:
        if not os.path.isdir(self.path):
            print('Ryn Directory not found')
            exit()

        self.split_dir.check()
        self.text_dir.check()
