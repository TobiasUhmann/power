"""
The `AnyBURL Directory` contains the `AnyBURL Facts Directory` that contains
the input `AnyBURL Dataset` for rule mining using AnyBURL as well as the
`AnyBURL Rules Directory` that contains the resulting, mined rules used
by the `POWER Ruler` later on.

**Structure**

::

    anyburl/                    # AnyBURL Directory

        facts/                  # AnyBURL Facts Directory

            cw_train_facts.tsv  # AnyBURL CW Train Facts TSV
            cw_valid_facts.tsv  # AnyBURL CW Valid Facts TSV
            ow_valid_facts.tsv  # AnyBURL OW Valid Facts TSV
            ow_test_facts.tsv   # AnyBURL OW Test Facts TSV

        rules/                  # AnyBURL Rules Directory

            cw_train_rules.tsv  # AnyBURL CW Train Rules TSV

|
"""

from pathlib import Path

from data.anyburl.facts.facts_dir import FactsDir
from data.anyburl.rules.rules_dir import RulesDir
from data.base_dir import BaseDir


class AnyburlDir(BaseDir):
    facts_dir: FactsDir
    rules_dir: RulesDir

    def __init__(self, path: Path):
        super().__init__(path)

        self.facts_dir = FactsDir(path.joinpath('facts'))
        self.rules_dir = RulesDir(path.joinpath('rules'))

    def check(self) -> None:
        super().check()

        self.facts_dir.check()
        self.rules_dir.check()
