"""
The `Neo4j Directory` contains the `Neo4j Dataset`, which is the KG defined
in the `IRT Split Directory`, but in a form that can be imported to Neo4j.

**Structure**

::

    neo4j/                  # Neo4j Directory

        entities.tsv        # Neo4j Entities TSV
        relations.tsv       # Neo4j Relations TSV

        cw_train_facts.tsv  # Neo4j CW Train Facts TSV
        cw_valid_facts.tsv  # Neo4j CW Valid Facts TSV
        ow_valid_facts.tsv  # Neo4j OW Valid Facts TSV
        ow_test_facts.tsv   # Neo4j OW Test Facts TSV

|
"""

from pathlib import Path

from data.base_dir import BaseDir
from data.neo4j.entities_tsv import EntitiesTsv
from data.neo4j.facts_tsv import FactsTsv
from data.neo4j.relations_tsv import RelationsTsv


class Neo4jDir(BaseDir):

    entities_tsv: EntitiesTsv
    relations_tsv: RelationsTsv

    cw_train_facts_tsv: FactsTsv
    cw_valid_facts_tsv: FactsTsv
    ow_valid_facts_tsv: FactsTsv
    ow_test_facts_tsv: FactsTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.entities_tsv = EntitiesTsv(path.joinpath('entities.tsv'))
        self.relations_tsv = RelationsTsv(path.joinpath('relations.tsv'))

        self.cw_train_facts_tsv = FactsTsv(path.joinpath('cw_train_facts.tsv'))
        self.cw_valid_facts_tsv = FactsTsv(path.joinpath('cw_valid_facts.tsv'))
        self.ow_valid_facts_tsv = FactsTsv(path.joinpath('ow_valid_facts.tsv'))
        self.ow_test_facts_tsv = FactsTsv(path.joinpath('ow_test_facts.tsv'))

    def check(self) -> None:
        super().check()

        self.entities_tsv.check()
        self.relations_tsv.check()

        self.cw_train_facts_tsv.check()
        self.cw_valid_facts_tsv.check()
        self.ow_valid_facts_tsv.check()
        self.ow_test_facts_tsv.check()
