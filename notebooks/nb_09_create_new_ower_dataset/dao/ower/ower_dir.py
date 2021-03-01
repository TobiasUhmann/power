"""
The `OWER Directory` contains the input files required for training the
`OWER Classifier`.

**Structure**

::

    ower/           # OWER Directory

        test.tsv    # OWER Samples TSV
        train.tsv   # OWER Samples TSV
        valid.tsv   # OWER Samples TSV

        test.db     # OWER Triples DB
        train.db    # OWER Triples DB
        valid.db    # OWER Triples DB

|
"""

from pathlib import Path

from dao.base_dir import BaseDir
from dao.ower.ower_samples_tsv import SamplesTsv
from dao.ower.ower_triples_db import TriplesDb


class OwerDir(BaseDir):

    train_triples_db: TriplesDb
    valid_triples_db: TriplesDb
    test_triples_db: TriplesDb

    train_samples_tsv: SamplesTsv
    valid_samples_tsv: SamplesTsv
    test_samples_tsv: SamplesTsv

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

        self.train_triples_db = TriplesDb('OWER Train Triples DB', path.joinpath('train.db'))
        self.valid_triples_db = TriplesDb('OWER Valid Triples DB', path.joinpath('valid.db'))
        self.test_triples_db = TriplesDb('OWER Test Triples DB', path.joinpath('test.db'))

        self.train_samples_tsv = SamplesTsv('OWER Train Samples TSV', path.joinpath('train.tsv'))
        self.valid_samples_tsv = SamplesTsv('OWER Valid Samples TSV', path.joinpath('valid.tsv'))
        self.test_samples_tsv = SamplesTsv('OWER Test Samples TSV', path.joinpath('test.tsv'))

    def check(self) -> None:
        super().check()

        self.train_triples_db.check()
        self.valid_triples_db.check()
        self.test_triples_db.check()

        self.train_samples_tsv.check()
        self.valid_samples_tsv.check()
        self.test_samples_tsv.check()
