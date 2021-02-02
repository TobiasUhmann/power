"""
The `Ryn Text Directory` contains the entities' sentences.

**Structure**

::

    text/                       # Ryn Text Directory

        cw.train-sentences.txt  # Ryn Sentences TXT
        ow.valid-sentences.txt  # Ryn Sentences TXT
        ow.test-sentences.txt   # Ryn Sentences TXT

|
"""

from pathlib import Path

from dao.base_dir import BaseDir
from dao.ryn.text.ryn_sentences_txt import RynSentencesTxt


class RynTextDir(BaseDir):

    _cw_train_sentences_txt: RynSentencesTxt
    _ow_valid_sentences_txt: RynSentencesTxt
    _ow_test_sentences_txt: RynSentencesTxt

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)
        
        self._cw_train_sentences_txt = RynSentencesTxt('Ryn CW Train Sentences TXT', path.joinpath('cw.train-sentences.txt'))
        self._ow_valid_sentences_txt = RynSentencesTxt('Ryn OW Valid Sentences TXT', path.joinpath('ow.valid-sentences.txt'))
        self._ow_test_sentences_txt = RynSentencesTxt('Ryn OW Test Sentences TXT', path.joinpath('ow.test-sentences.txt'))

    def check(self) -> None:
        super().check()

        self._cw_train_sentences_txt.check()
        self._ow_valid_sentences_txt.check()
        self._ow_test_sentences_txt.check()
