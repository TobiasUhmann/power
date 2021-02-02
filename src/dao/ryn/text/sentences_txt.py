from collections import defaultdict
from os.path import isfile
from pathlib import Path
from typing import Dict, Set


class SentencesTxt:
    name: str
    path: Path

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

    def check(self) -> None:
        if not isfile(self.path):
            print(f'{self.name} not found')
            exit()

    def load_ent_to_sentences(self) -> Dict[int, Set[str]]:
        ent_to_contexts: Dict[int, Set[str]] = defaultdict(set)

        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]:
            ent, _, context = line.split(' | ')
            ent_to_contexts[int(ent)].add(context.strip())

        return ent_to_contexts
