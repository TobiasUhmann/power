"""
The `AnyBURL Triples TXT` contains the input triples for AnyBURL
rule mining. It has the structure required by AnyBURL
(http://web.informatik.uni-mannheim.de/AnyBURL/):

* 3 columns, no header row
* Tabular separated values
* No values that can be parsed as an integer

**Example**

::

    Stan_Collymore	playsFor	England_national_football_team
    Suriname	hasOfficialLanguage	Dutch_language
    Mantorras	playsFor	F.C._Alverca

|
"""

from pathlib import Path
from typing import List, Tuple

from dao.base_file import BaseFile


class AnyburlTriplesTxt(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    def save_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        with open(self._path, 'w', encoding='utf-8') as f:
            for head, rel, tail in triples:
                f.write(f'{head}\t{rel}\t{tail}\n')
