from dataclasses import dataclass
from pathlib import Path
from typing import List

from data.base_file import BaseFile


@dataclass
class Relation:
    rel: int
    rel_lbl: str

    def __iter__(self):
        return iter((self.rel, self.rel_lbl))


class RelationsTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, relations: List[Relation]) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            row = '\t'.join(['{}'] * 2) + '\n'

            f.write(row.format('rel', 'rel_lbl'))
            for rel, rel_lbl in relations:
                f.write(row.format(rel, rel_lbl))
