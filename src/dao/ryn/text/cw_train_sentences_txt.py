from os.path import isfile
from pathlib import Path

from dao.ryn.text.sentences_txt import SentencesTxt


class CwTrainSentencesTxt(SentencesTxt):
    path: Path

    def __init__(self, path: Path):
        super().__init__(path)

    def check(self) -> None:
        if not isfile(self.path):
            print('CW Train Sentences TXT not found')
            exit()
