"""
The `AnyBURL Directory` contains the `AnyBURL Dataset`, which is the KG defined
in the `Ryn Split Directory`, but in a form that can be read by AnyBURL.

**Structure**

::

    anyburl/                # AnyBURL Directory

        cw_train_facts.tsv  # AnyBURL CW Train Facts TSV
        cw_valid_facts.tsv  # AnyBURL CW Valid Facts TSV
        ow_valid_facts.tsv  # AnyBURL OW Valid Facts TSV
        ow_test_facts.tsv   # AnyBURL OW Test Facts TSV

|
"""

from pathlib import Path

from data.anyburl.facts_tsv import FactsTsv
from data.base_dir import BaseDir


class AnyburlDir(BaseDir):

    cw_train_facts_tsv: FactsTsv
    cw_valid_facts_tsv: FactsTsv
    ow_valid_facts_tsv: FactsTsv
    ow_test_facts_tsv: FactsTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.cw_train_facts_tsv = FactsTsv(path.joinpath('cw_train_facts.tsv'))
        self.cw_valid_facts_tsv = FactsTsv(path.joinpath('cw_valid_facts.tsv'))
        self.ow_valid_facts_tsv = FactsTsv(path.joinpath('ow_valid_facts.tsv'))
        self.ow_test_facts_tsv = FactsTsv(path.joinpath('ow_test_facts.tsv'))

    def check(self) -> None:
        super().check()

        self.cw_train_facts_tsv.check()
        self.cw_valid_facts_tsv.check()
        self.ow_valid_facts_tsv.check()
        self.ow_test_facts_tsv.check()
