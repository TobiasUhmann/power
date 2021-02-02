"""
Provides functions to load/save a Samples TSV.
"""

from os.path import isfile
from pathlib import Path
from typing import List


class SamplesTsv:
    name: str
    path: Path

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

    def check(self) -> None:
        if not isfile(self.path):
            print(f'{self.name} not found')
            exit()

    def write_samples_tsv(self, rows: List) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            for row in rows:
                ent = row[0]
                classes = row[1:-1]
                sentences = row[-1]

                f.write(str(ent))
                for class_ in classes:
                    f.write(f'\t{str(class_)}')
                for sentence in sentences:
                    f.write(f'\t{sentence}')
                f.write('\n')
