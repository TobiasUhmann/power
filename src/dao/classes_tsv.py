from typing import List, Tuple


def load_classes(classes_tsv: str) -> List[Tuple[int, int]]:
    classes: List[Tuple[int, int]] = []

    with open(classes_tsv, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        rel, tail = line.split('\t')
        classes.append((int(rel), int(tail)))

    return classes
