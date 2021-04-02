"""
The `POWER Labels TSV` contains the entities' or relations' labels.

**Example**

::

    0   Dominican Republic
    1   republic
    2   Mighty Morphin Power Rangers

|
"""

import csv
from pathlib import Path
from typing import Dict

from data.base_file import BaseFile


class LabelsTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def save(self, ent_to_lbl: Dict[int, str]) -> None:
        with open(self.path, 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            csv_writer.writerow(('id', 'lbl'))

            for ent, lbl in ent_to_lbl.items():
                csv_writer.writerow((ent, lbl))

    def load(self) -> Dict[int, str]:
        with open(self.path, encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader)

            ent_to_lbl = {int(ent): lbl for ent, lbl in csv_reader}

        return ent_to_lbl
