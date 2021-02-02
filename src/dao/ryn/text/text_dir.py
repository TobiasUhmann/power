"""
Functions for checking the file structure of a Ryn Text Directory
"""
from os import makedirs
from os.path import isdir
from pathlib import Path

from dao.ryn.text.sentences_txt import SentencesTxt


class TextDir:
    name: str
    path: Path

    cw_train_sentences_txt: SentencesTxt
    ow_valid_sentences_txt: SentencesTxt
    ow_test_sentences_txt: SentencesTxt

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        
        self.cw_train_sentences_txt = SentencesTxt('CW Train Sentences TXT', path.joinpath('cw.train-sentences.txt'))
        self.ow_valid_sentences_txt = SentencesTxt('OW Valid Sentences TXT', path.joinpath('ow.valid-sentences.txt'))
        self.ow_test_sentences_txt = SentencesTxt('OW Test Sentences TXT', path.joinpath('ow.test-sentences.txt'))

    def check(self) -> None:
        if not isdir(self.path):
            print(f'{self.name} not found')
            exit()

        self.cw_train_sentences_txt.check()
        self.ow_valid_sentences_txt.check()
        self.ow_test_sentences_txt.check()

    def create(self) -> None:
        makedirs(self.path, exist_ok=True)
