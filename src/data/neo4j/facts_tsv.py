from dataclasses import dataclass
from pathlib import Path
from typing import List

from data.base_file import BaseFile


@dataclass
class Fact:
    head: int
    head_lbl: str

    rel: int
    rel_lbl: str

    tail: int
    tail_lbl: str

    def __iter__(self):
        return iter((self.head, self.head_lbl, self.rel, self.rel_lbl, self.tail, self.tail_lbl))


class FactsTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, facts: List[Fact]) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            row = '\t'.join(['{}'] * 6) + '\n'

            f.write(row.format('head', 'head_lbl', 'rel', 'rel_lbl', 'tail', 'tail_lbl'))
            for head, head_lbl, rel, rel_lbl, tail, tail_lbl in facts:
                f.write(row.format(head, head_lbl, rel, rel_lbl, tail, tail_lbl))
