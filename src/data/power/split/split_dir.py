"""
The `POWER Split Directory` contains the files that define the triples
split into train/valid/test.

**Structure**

::

    split/

        ent_labels.tsv
        rel_labels.tsv

        train_facts.tsv
        
        valid_facts_25_1.tsv
        valid_facts_25_2.tsv
        valid_facts_25_3.tsv
        valid_facts_25_4.tsv
        
        test_facts_25_1.tsv
        test_facts_25_2.tsv
        test_facts_25_3.tsv
        test_facts_25_4.tsv

|
"""

from pathlib import Path

from data.base_dir import BaseDir
from data.power.split.facts_tsv import FactsTsv
from data.power.split.labels_tsv import LabelsTsv


class SplitDir(BaseDir):

    ent_labels_tsv: LabelsTsv
    rel_labels_tsv: LabelsTsv
    
    train_facts_tsv: FactsTsv
    
    valid_facts_25_1_tsv: FactsTsv
    valid_facts_25_2_tsv: FactsTsv
    valid_facts_25_3_tsv: FactsTsv
    valid_facts_25_4_tsv: FactsTsv
    
    test_facts_25_1_tsv: FactsTsv
    test_facts_25_2_tsv: FactsTsv
    test_facts_25_3_tsv: FactsTsv
    test_facts_25_4_tsv: FactsTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.ent_labels_tsv = LabelsTsv(path.joinpath('ent_labels.tsv'))
        self.rel_labels_tsv = LabelsTsv(path.joinpath('rel_labels.tsv'))

        self.train_facts_tsv = FactsTsv(path.joinpath('train_facts.tsv'))

        self.valid_facts_25_1_tsv = FactsTsv(path.joinpath('valid_facts_25_1.tsv'))
        self.valid_facts_25_2_tsv = FactsTsv(path.joinpath('valid_facts_25_2.tsv'))
        self.valid_facts_25_3_tsv = FactsTsv(path.joinpath('valid_facts_25_3.tsv'))
        self.valid_facts_25_4_tsv = FactsTsv(path.joinpath('valid_facts_25_4.tsv'))

        self.test_facts_25_1_tsv = FactsTsv(path.joinpath('test_facts_25_1.tsv'))
        self.test_facts_25_2_tsv = FactsTsv(path.joinpath('test_facts_25_2.tsv'))
        self.test_facts_25_3_tsv = FactsTsv(path.joinpath('test_facts_25_3.tsv'))
        self.test_facts_25_4_tsv = FactsTsv(path.joinpath('test_facts_25_4.tsv'))

    def check(self) -> None:
        super().check()

        self.ent_labels_tsv.check()
        self.rel_labels_tsv.check()

        self.train_facts_tsv.check()

        self.valid_facts_25_1_tsv.check()
        self.valid_facts_25_2_tsv.check()
        self.valid_facts_25_3_tsv.check()
        self.valid_facts_25_4_tsv.check()

        self.test_facts_25_1_tsv.check()
        self.test_facts_25_2_tsv.check()
        self.test_facts_25_3_tsv.check()
        self.test_facts_25_4_tsv.check()
