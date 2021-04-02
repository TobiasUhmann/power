"""
The `Neo4j Directory` contains the `Neo4j Dataset`, which is the KG defined
in the `IRT Split Directory`, but in a form that can be imported to Neo4j.

**Structure**

::

    neo4j/                    # Neo4j Directory

        entities.tsv          # Neo4j Entities TSV
        relations.tsv         # Neo4j Relations TSV
        
        train-facts.tsv       # Neo4j Train Facts TSV
        
        valid-facts_25-1.tsv  # Neo4j Valid Facts 25-1 TSV
        valid-facts_25-2.tsv  # Neo4j Valid Facts 25-2 TSV
        valid-facts_25-3.tsv  # Neo4j Valid Facts 25-3 TSV
        valid-facts_25-4.tsv  # Neo4j Valid Facts 25-4 TSV
        
        test-facts_25-1.tsv   # Neo4j Test Facts 25-1 TSV
        test-facts_25-2.tsv   # Neo4j Test Facts 25-2 TSV
        test-facts_25-3.tsv   # Neo4j Test Facts 25-3 TSV
        test-facts_25-4.tsv   # Neo4j Test Facts 25-4 TSV

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

    train_facts_tsv: FactsTsv

    valid_facts_25_1_tsv: FactsTsv
    valid_facts_25_1_tsv: FactsTsv
    valid_facts_25_1_tsv: FactsTsv
    valid_facts_25_1_tsv: FactsTsv

    test_facts_25_1_tsv: FactsTsv
    test_facts_25_1_tsv: FactsTsv
    test_facts_25_1_tsv: FactsTsv
    test_facts_25_1_tsv: FactsTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.entities_tsv = EntitiesTsv(path.joinpath('entities.tsv'))
        self.relations_tsv = RelationsTsv(path.joinpath('relations.tsv'))

        self.train_facts_tsv = FactsTsv(path.joinpath('train-facts.tsv'))

        self.valid_facts_25_1_tsv = FactsTsv(path.joinpath('valid-facts_25-1.tsv'))
        self.valid_facts_25_2_tsv = FactsTsv(path.joinpath('valid-facts_25-2.tsv'))
        self.valid_facts_25_3_tsv = FactsTsv(path.joinpath('valid-facts_25-3.tsv'))
        self.valid_facts_25_4_tsv = FactsTsv(path.joinpath('valid-facts_25-4.tsv'))

        self.test_facts_25_1_tsv = FactsTsv(path.joinpath('test-facts_25-1.tsv'))
        self.test_facts_25_2_tsv = FactsTsv(path.joinpath('test-facts_25-2.tsv'))
        self.test_facts_25_3_tsv = FactsTsv(path.joinpath('test-facts_25-3.tsv'))
        self.test_facts_25_4_tsv = FactsTsv(path.joinpath('test-facts_25-4.tsv'))

    def check(self) -> None:
        super().check()

        self.entities_tsv.check()
        self.relations_tsv.check()

        self.train_facts_tsv.check()

        self.valid_facts_25_1_tsv.check()
        self.valid_facts_25_2_tsv.check()
        self.valid_facts_25_3_tsv.check()
        self.valid_facts_25_4_tsv.check()

        self.test_facts_25_1_tsv.check()
        self.test_facts_25_2_tsv.check()
        self.test_facts_25_3_tsv.check()
        self.test_facts_25_4_tsv.check()
