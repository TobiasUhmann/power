from os.path import isfile
from pathlib import Path

from dao.ryn.split.labels_txt import LabelsTxt


class EntityLabelsTxt(LabelsTxt):
    path: Path

    def __init__(self, path: Path):
        super().__init__(path)

    def check(self) -> None:
        if not isfile(self.path):
            print('Entity Labels TXT not found')
            exit()
