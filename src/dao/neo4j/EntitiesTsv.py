from dataclasses import dataclass
from pathlib import Path
from typing import List

from dao.base_file import BaseFile


@dataclass
class Entity:
    ent: int
    ent_lbl: str

    def __iter__(self):
        return iter((self.ent, self.ent_lbl))


class EntitiesTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, entities: List[Entity]) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            row = '\t'.join(['{}'] * 2) + '\n'

            f.write(row.format('ent', 'ent_lbl'))
            for ent, ent_lbl in entities:
                f.write(row.format(ent, ent_lbl))
