"""
The `Ryn Triples TXT` stores triples using RIDs.

* Header row specifies number of triples
* Space separated RIDs
* Head - Tail - Relation

**Example**

::

    214412
    10195 7677 22
    4253 450 69
    5806 2942 32
    2271 6322 203

|
"""

from pathlib import Path
from typing import List, Tuple

from dao.base_file import BaseFile


class RynTriplesTxt(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    def load_triples(self) -> List[Tuple[int, int, int]]:
        with open(self._path, encoding='utf-8') as f:
            lines = f.readlines()

        triples: List[Tuple[int, int, int]] = []

        for line in lines[1:]:
            head, tail, rel = line.split()
            triples.append((int(head), int(rel), int(tail)))

        return triples
