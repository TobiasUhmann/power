"""
The `AnyBURL Facts TSV` contains the input facts for AnyBURL
rule mining. It has the structure required by AnyBURL
(http://web.informatik.uni-mannheim.de/AnyBURL/):

* Tabular separated values, no header
* 3 columns: Head entity (String), Relation (String), Tail entity (String)
* Relation must not be parsable as an integer

**Example**

::

    Stan_Collymore	playsFor	England_national_football_team
    Suriname	hasOfficialLanguage	Dutch_language
    Mantorras	playsFor	F.C._Alverca

|
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from data.base_file import BaseFile


@dataclass
class Fact:
    head: str
    rel: str
    tail: str

    def __iter__(self):
        return iter((self.head, self.rel, self.tail))


class FactsTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, facts: List[Fact]) -> None:
        with open(self.path, 'w', encoding='utf-8') as f:
            row = '\t'.join(['{}'] * 3) + '\n'

            f.write(row.format('head', 'rel', 'tail'))
            for head, rel, tail in facts:
                f.write(row.format(head, rel, tail))
