"""
Functions for checking the file structure of a Ryn Text Directory
"""

from os.path import isdir
from pathlib import Path

from dao.ryn.text.ow_test_sentences_txt import OwTestSentencesTxt
from dao.ryn.text.cw_train_sentences_txt import CwTrainSentencesTxt
from dao.ryn.text.ow_valid_sentences_txt import OwValidSentencesTxt


class TextDir:
    path: Path

    cw_train_sentences_txt: CwTrainSentencesTxt
    ow_valid_sentences_txt: OwValidSentencesTxt
    ow_test_sentences_txt: OwTestSentencesTxt

    def __init__(self, path: Path):
        self.path = path
        
        self.cw_train_sentences_txt = CwTrainSentencesTxt(path.joinpath('cw.train-sentences.txt'))
        self.ow_valid_sentences_txt = OwValidSentencesTxt(path.joinpath('ow.valid-sentences.txt'))
        self.ow_test_sentences_txt = OwTestSentencesTxt(path.joinpath('ow.test-sentences.txt'))

    def check(self) -> None:
        if not isdir(self.path):
            print('Text Directory not found')
            exit()

        self.cw_train_sentences_txt.check()
        self.ow_valid_sentences_txt.check()
        self.ow_test_sentences_txt.check()
