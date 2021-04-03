"""
The `POWER Split Directory` contains the files that define the triples
split into train/valid/test.

**Structure**

::

    split/

        entities.tsv
        relations.tsv

        train.tsv
        
        valid_known.tsv
        valid_unknown.tsv

        test_known.tsv
        test_unknown.tsv

|
"""

from pathlib import Path

from data.base_dir import BaseDir
from data.power.split.facts_tsv import FactsTsv
from data.power.split.labels_tsv import LabelsTsv


class SplitDir(BaseDir):

    entities_tsv: LabelsTsv
    relations_tsv: LabelsTsv
    
    train_tsv: FactsTsv
    
    valid_known_tsv: FactsTsv
    valid_unknown_tsv: FactsTsv

    test_known_tsv: FactsTsv
    test_unknown_tsv: FactsTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.entities_tsv = LabelsTsv(path.joinpath('entities.tsv'))
        self.relations_tsv = LabelsTsv(path.joinpath('relations.tsv'))

        self.train_tsv = FactsTsv(path.joinpath('train.tsv'))

        self.valid_known_tsv = FactsTsv(path.joinpath('valid_known.tsv'))
        self.valid_unknown_tsv = FactsTsv(path.joinpath('valid_unknown.tsv'))

        self.test_known_tsv = FactsTsv(path.joinpath('test_known.tsv'))
        self.test_unknown_tsv = FactsTsv(path.joinpath('test_unknown.tsv'))

    def check(self) -> None:
        super().check()

        self.entities_tsv.check()
        self.relations_tsv.check()

        self.train_tsv.check()

        self.valid_known_tsv.check()
        self.valid_unknown_tsv.check()

        self.test_known_tsv.check()
        self.test_unknown_tsv.check()
