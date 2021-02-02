"""
The `AnyBURL Directory` contains the triple files that AnyBURL uses
as input to mine rules.

**Structure**

::

    anyburl/        # AnyBURL Directory

        train.txt   # AnyBURL Triples TXT
        valid.txt   # AnyBURL Triples TXT
        test.txt    # AnyBURL Triples TXT

|
"""

from pathlib import Path

from dao.anyburl.anyburl_triples_txt import AnyburlTriplesTxt
from dao.base_dir import BaseDir


class AnyburlDir(BaseDir):
    
    _train_triples_txt: AnyburlTriplesTxt
    _valid_triples_txt: AnyburlTriplesTxt
    _test_triples_txt: AnyburlTriplesTxt

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)
        
        self._train_triples_txt = AnyburlTriplesTxt('AnyBURL Train Triples TXT', path.joinpath('train.txt'))
        self._valid_triples_txt = AnyburlTriplesTxt('AnyBURL Valid Triples TXT', path.joinpath('valid.txt'))
        self._test_triples_txt = AnyburlTriplesTxt('AnyBURL Test Triples TXT', path.joinpath('test.txt'))

    def check(self) -> None:
        super().check()

        self._train_triples_txt.check()
        self._valid_triples_txt.check()
        self._test_triples_txt.check()
