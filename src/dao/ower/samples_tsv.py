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
from typing import List, Tuple

from dao.base_file import BaseFile


class SamplesTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, ent_lbl_classes_sents_list: List[Tuple[int, str, List[int], List[str]]]) -> None:
        """
        :param ent_lbl_classes_sents_list: [(ent, label, [has class], [sent]]
        """

        with open(self.path, 'w', encoding='utf-8') as f:
            for ent, label, classes, sents in ent_lbl_classes_sents_list:
                f.write('{:6}\t{:40}\t{}\t  {}\n'.format(
                    ent, label, '\t'.join((str(c) for c in classes)), '\t    '.join(sents)))
