from dataclasses import dataclass
from pathlib import Path
from typing import List

from dao.base_file import BaseFile


@dataclass
class Fact:
    head: int
    rel: int
    tail: int


class FactsTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, facts: List[Fact]) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('{:6}\t{:6}\t{:6}\n'.format('Head', 'Rel', 'Tail'))

            for fact in facts:
                f.write('{:6}\t{:6}\t{:6}\n'.format(fact.head, fact.rel, fact.tail))
