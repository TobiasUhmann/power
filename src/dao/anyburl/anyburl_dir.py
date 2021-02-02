"""
The `AnyBURL Directory` contains the triple files that AnyBURL uses
as input to mine rules.

Structure::

    anyburl/        # AnyBURL Dir, v1
        test.txt    # AnyBURL Triples TXT, v1
        train.txt   # AnyBURL Triples TXT, v1
        valid.txt   # AnyBURL Triples TXT, v1

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
        
        self._train_triples_txt = AnyburlTriplesTxt('AnyBURL Train Triples TXT', self._path)
        self._valid_triples_txt = AnyburlTriplesTxt('AnyBURL Valid Triples TXT', self._path)
        self._test_triples_txt = AnyburlTriplesTxt('AnyBURL Test Triples TXT', self._path)

    def check(self) -> None:
        super().check()

        self._train_triples_txt.check()
        self._valid_triples_txt.check()
        self._test_triples_txt.check()
