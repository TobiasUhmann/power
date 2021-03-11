"""
The `OWER Classes TSV` gives detailed information about the classes used in
the `OWER Samples TSV`s.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dao.base_file import BaseFile


class ClassesTsv(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    @dataclass
    class Row:
        rel: int
        tail: int
        freq: float
        label: str

    def save(self, rows: List[Row]) -> None:
        with open(self._path, 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            csv_writer.writerow(('rel', 'tail', 'freq', 'label'))

            for row in rows:
                csv_writer.writerow((row.rel, row.tail, row.freq, row.label))

    def load(self) -> List[Row]:
        with open(self._path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            csv_reader.next()

            rows = []
            for rel, tail, freq, label in csv_reader:
                rel = int(rel)
                tail = int(tail)
                freq = float(freq)

                rows.append(ClassesTsv.Row(rel, tail, freq, label))

        return rows
