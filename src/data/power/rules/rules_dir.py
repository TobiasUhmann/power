"""
The `POWER Rules Directory` contains the `rules.tsv`, which corresponds to one
of the rules files created by AnyBURL, in adition to the entity/relation labels TXTs.

**Structure**

::

    rules/

        rules.tsv

        ent_labels.txt
        rel_labels.txt

|
"""

from pathlib import Path

from data.power.rules.rules_tsv import RulesTsv
from data.base_dir import BaseDir
from data.irt.split.labels_txt import LabelsTxt


class RulesDir(BaseDir):

    rules_tsv: RulesTsv

    ent_labels_txt: LabelsTxt
    rel_labels_txt: LabelsTxt

    def __init__(self, path: Path):
        super().__init__(path)

        self.rules_tsv = RulesTsv(path.joinpath('rules.tsv'))

        self.ent_labels_txt = LabelsTxt(path.joinpath('ent_labels.txt'))
        self.rel_labels_txt = LabelsTxt(path.joinpath('rel_labels.txt'))

    def check(self) -> None:
        super().check()

        self.rules_tsv.check()

        self.ent_labels_txt.check()
        self.rel_labels_txt.check()
