"""
The `OWER Samples TSV` contains the input data for training the
`OWER Classifier`.

* Tabular separated
* 1 Header Row
* First column: Entity RID
* N columns: class_1 .. class_n
* M sentences

**Example**

::

    entity  class_1 class_2 class_3 class_4 sent_1  sent_2  sent_3
    1   0	0	0	0	Foo.    Bar.    Baz.
    2   0	1	0	1	Lorem.  Ypsum.  Dolor

|
"""

from pathlib import Path
from typing import List

from dao.base_file import BaseFile


class SamplesTsv(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    def write_samples_tsv(self, rows: List) -> None:
        with open(self._path, 'w', encoding='utf-8') as f:
            for row in rows:
                ent = row[0]
                classes = row[1:-1]
                sentences = row[-1]

                f.write(str(ent))
                for class_ in classes:
                    f.write(f'\t{str(class_)}')
                for sentence in sentences:
                    f.write(f'\t{sentence}')
                f.write('\n')
