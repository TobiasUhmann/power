"""
The `Ryn Labels TXT` contains the entities' or relations' labels.

* Header row specifies number of entities / relations
* Space separated

**Example**

::

    14541
    Dominican Republic 0
    republic 1
    Mighty Morphin Power Rangers 2

|
"""
import logging
from pathlib import Path
from typing import Dict

from dao.base_file import BaseFile


class RynLabelsTxt(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    def load_rid_to_label(self) -> Dict[int, str]:
        """
        :return: RID -> Label mapping
        """

        with open(self._path, encoding='utf-8') as f:
            lines = f.readlines()

        rid_to_label: Dict[int, str] = {}

        for line in lines[1:]:
            parts = line.split()

            parts_by_space = line.split(' ')
            if len(parts) != len(parts_by_space):
                logging.warning('[WARN] Line must contain single spaces only as separator.'
                                f' Replacing each whitespace with single space. Line: {repr(line)}')

            label = ' '.join(parts[:-1])
            rid = int(parts[-1])

            rid_to_label[rid] = label

        return rid_to_label
