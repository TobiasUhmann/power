import pickle
from pathlib import Path

from data.base_file import BaseFile
from power.ruler import Ruler


class RulerPkl(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, ruler: Ruler) -> None:
        with open(self.path, 'wb') as f:
            pickle.dump(ruler, f)

    def load(self) -> Ruler:
        with open(self.path, 'rb') as f:
            return pickle.load(f)
