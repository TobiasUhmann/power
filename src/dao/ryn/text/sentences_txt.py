from collections import defaultdict
from pathlib import Path
from typing import Dict, Set


class SentencesTxt:
    path: Path

    def __init__(self, path: Path):
        self.path = path

    def check(self) -> None:
        pass

    def load_ent_to_sentences(self) -> Dict[int, Set[str]]:
        ent_to_contexts: Dict[int, Set[str]] = defaultdict(set)

        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]:
            ent, _, context = line.split(' | ')
            ent_to_contexts[int(ent)].add(context.strip())

        return ent_to_contexts
