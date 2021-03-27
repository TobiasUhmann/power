from dataclasses import dataclass
from pathlib import Path
from typing import List

from dao.base_file import BaseFile


@dataclass
class Entity:
    id: int
    label: str


class EntitiesTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, entities: List[Entity]) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('{:6}\t{}\n'.format('ID', 'Label'))

            for entity in entities:
                f.write('{:6}\t{}\n'.format(entity.id, entity.label))
