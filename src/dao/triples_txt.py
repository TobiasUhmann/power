from typing import List, Tuple


def load_triples(triples_txt: str) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []

    with open(triples_txt, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        head, tail, rel = line.split()
        triples.append((int(head), int(rel), int(tail)))

    return triples
