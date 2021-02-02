from os.path import isfile
from pathlib import Path

from dao.ryn.split.labels_txt import LabelsTxt


class RelationLabelsTxt(LabelsTxt):
    path: Path

    def __init__(self, path: Path):
        super().__init__(path)

    def check(self) -> None:
        if not isfile(self.path):
            print('Relation Labels TXT not found')
            exit()
