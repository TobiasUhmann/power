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

from dao.anyburl.triples_txt import TriplesTxt
from dao.base_dir import BaseDir


class AnyburlDir(BaseDir):
    
    train_triples_txt: TriplesTxt
    valid_triples_txt: TriplesTxt
    test_triples_txt: TriplesTxt

    def __init__(self, path: Path):
        super().__init__(path)
        
        self.train_triples_txt = TriplesTxt(path.joinpath('train.txt'))
        self.valid_triples_txt = TriplesTxt(path.joinpath('valid.txt'))
        self.test_triples_txt = TriplesTxt(path.joinpath('test.txt'))

    def check(self) -> None:
        super().check()

        self.train_triples_txt.check()
        self.valid_triples_txt.check()
        self.test_triples_txt.check()
