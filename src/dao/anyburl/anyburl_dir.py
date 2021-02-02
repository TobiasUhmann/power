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
    
    train_triples_txt: AnyburlTriplesTxt
    valid_triples_txt: AnyburlTriplesTxt
    test_triples_txt: AnyburlTriplesTxt

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)
        
        self.train_triples_txt = AnyburlTriplesTxt('AnyBURL Train Triples TXT', path.joinpath('train.txt'))
        self.valid_triples_txt = AnyburlTriplesTxt('AnyBURL Valid Triples TXT', path.joinpath('valid.txt'))
        self.test_triples_txt = AnyburlTriplesTxt('AnyBURL Test Triples TXT', path.joinpath('test.txt'))

    def check(self) -> None:
        super().check()

        self.train_triples_txt.check()
        self.valid_triples_txt.check()
        self.test_triples_txt.check()
