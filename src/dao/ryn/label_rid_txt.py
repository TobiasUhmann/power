import re
from typing import Dict


def load_rid_to_label(label_rid_txt: str) -> Dict[int, str]:
    """
    :param label_rid_txt: path to 'Label-RID TXT'
    :return: RID -> Label mapping
    """

    with open(label_rid_txt, encoding='utf-8') as f:
        lines = f.readlines()

    rid_to_label: Dict[int, str] = {}

    for line in lines[1:]:
        parts = line.split()

        parts_by_space = line.split(' ')
        if len(parts) != len(parts_by_space):
            print('[WARN] Line must contain single spaces only as separator.'
                  f' Replacing each whitespace with single space. Line: {repr(line)}')

        label = ' '.join(parts[:-1])
        rid = int(parts[-1])

        rid_to_label[rid] = label

    return rid_to_label
