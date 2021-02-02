from os.path import isfile
from pathlib import Path
from typing import Dict


class LabelsTxt:
    name: str
    path: Path

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

    def check(self) -> None:
        if not isfile(self.path):
            print(f'{self.name} not found')
            exit()

    def load_rid_to_label(self) -> Dict[int, str]:
        """
        :return: RID -> Label mapping
        """

        with open(self.path, encoding='utf-8') as f:
            lines = f.readlines()

        rid_to_label: Dict[int, str] = {}

        for line in lines[1:]:
            parts = line.split()

            parts_by_space = line.split(' ')
            if len(parts) != len(parts_by_space):
                print('[WARN] Line must contain single spaces only as separator.'
                      f' Replacing each whitespace with single space. Line: {repr(line)}')

            label = ' '.join(parts[:-1])
            rid = int(parts[-1])

            rid_to_label[rid] = label

        return rid_to_label
