from os.path import isfile
from pathlib import Path

from dao.ryn.text.sentences_txt import SentencesTxt


class OwValidSentencesTxt(SentencesTxt):
    path: Path

    def __init__(self, path: Path):
        super().__init__(path)

    def check(self) -> None:
        if not isfile(self.path):
            print('OW Valid Sentences TXT not found')
            exit()
