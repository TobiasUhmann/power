from dataclasses import dataclass
from pathlib import Path
from typing import List

from dao.base_file import BaseFile


@dataclass
class Relation:
    id: int
    label: str


class RelationsTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, relations: List[Relation]) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('{:6}\t{}\n'.format('ID', 'Label'))

            for relation in relations:
                f.write('{:6}\t{}\n'.format(relation.id, relation.label))
