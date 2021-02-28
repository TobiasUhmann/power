"""
The `Classes TSV` contains the relation-tail tuples that make up
the output classes of the `OWER Classifier`.

* Tabular separated
* 1 Header Row
* First column: Relation RID
* Second column: Tail entity RID

**Example**

::

    rel tail
    43  141
    85	434
    48	32
    17	862

|
"""

from pathlib import Path
from typing import List, Tuple

from dao.base_file import BaseFile


class ClassesTsv(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    def read_classes(classes_tsv: str) -> List[Tuple[int, int]]:
        classes: List[Tuple[int, int]] = []

        with open(classes_tsv, encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]:
            rel, tail = line.split('\t')
            classes.append((int(rel), int(tail)))

        return classes
