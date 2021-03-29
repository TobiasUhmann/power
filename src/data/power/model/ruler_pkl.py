import pickle
from pathlib import Path
from typing import Tuple, Dict, List

from data.base_file import BaseFile
from models.ent import Ent
from models.rel import Rel
from models.rule import Rule


class RulerPkl(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, ruler: Dict[Ent, Dict[Tuple[Rel, Ent], List[Rule]]]) -> None:
        with open(self.path, 'wb') as f:
            pickle.dump(ruler, f)

    def load(self) -> Dict[Ent, Dict[Tuple[Rel, Ent], List[Rule]]]:
        with open(self.path, 'rb') as f:
            return pickle.load(f)
