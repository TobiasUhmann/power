from os.path import isfile
from pathlib import Path

from dao.ryn.split.triples_txt import TriplesTxt


class CwTrainTriplesTxt(TriplesTxt):
    path: Path

    def __init__(self, path: Path):
        super().__init__(path)

    def check(self) -> None:
        if not isfile(self.path):
            print('CW Train Triples TXT not found')
            exit()
