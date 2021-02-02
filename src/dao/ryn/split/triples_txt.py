from os.path import isfile
from pathlib import Path
from typing import List, Tuple


class TriplesTxt:
    name: str
    path: Path

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

    def check(self) -> None:
        if not isfile(self.path):
            print(f'{self.name} not found')
            exit()

    def load_triples(self) -> List[Tuple[int, int, int]]:
        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()

        triples: List[Tuple[int, int, int]] = []

        for line in lines[1:]:
            head, tail, rel = line.split()
            triples.append((int(head), int(rel), int(tail)))

        return triples
