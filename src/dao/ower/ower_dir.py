"""
The `OWER Directory` contains the input files required for training the
`OWER Classifier`.

==
v3
==

* Incorporates `Triples DBs` from former `Work Directory`

::

    ower-v3/
        samples-v2-test.tsv
        samples-v2-train.tsv
        samples-v2-valid.tsv
        triples-v1-test.db
        triples-v1-train.db
        triples-v1-valid.db

==
v2
==

* Upgraded `Samples TXTs` to v2

::

    ower-v2/
        samples-v2-test.tsv
        samples-v2-train.tsv
        samples-v2-valid.tsv

==
v1
==

::

    ower-v1/
        samples-v1-test.tsv
        samples-v1-train.tsv
        samples-v1-valid.tsv

"""
from os import makedirs
from os.path import isdir
from pathlib import Path

from dao.ower.samples_tsv import SamplesTsv
from dao.ower.triples_db import TriplesDb


class OwerDir:
    name: str
    path: Path

    tain_triples_db: TriplesDb
    valid_triples_db: TriplesDb
    test_triples_db: TriplesDb

    train_samples_tsv: SamplesTsv
    valid_samples_tsv: SamplesTsv
    test_samples_tsv: SamplesTsv
    
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        
        self.train_triples_db = TriplesDb('Train Triples DB', path.joinpath('triples-v1-train.db'))
        self.valid_triples_db = TriplesDb('Valid Triples DB', path.joinpath('triples-v1-valid.db'))
        self.test_triples_db = TriplesDb('Test Triples DB', path.joinpath('triples-v1-test.db'))
        
        self.train_samples_tsv = SamplesTsv('Train Samples TSV', path.joinpath('samples-v2-train.tsv'))
        self.valid_samples_tsv = SamplesTsv('Valid Samples TSV', path.joinpath('samples-v2-valid.tsv'))
        self.test_samples_tsv = SamplesTsv('Test Samples TSV', path.joinpath('samples-v2-test.tsv'))

    def check(self) -> None:
        if not isdir(self.path):
            print(f'{self.name} not found')
            exit()

        self.train_triples_db.check()
        self.valid_triples_db.check()
        self.test_triples_db.check()

        self.train_samples_tsv.check()
        self.valid_samples_tsv.check()
        self.test_samples_tsv.check()

    def create(self) -> None:
        makedirs(self.path, exist_ok=True)
