from os.path import isfile
from pathlib import Path

from dao.ryn.split.triples_txt import TriplesTxt


class OwValidTriplesTxt(TriplesTxt):
    path: Path

    def __init__(self, path: Path):
        super().__init__(path)

    def check(self) -> None:
        if not isfile(self.path):
            print('OW Valid Triples TXT not found')
            exit()
