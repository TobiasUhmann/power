from typing import List, Tuple


def save(triples_txt: str, triples: List[Tuple[str, str, str]]) -> None:
    with open(triples_txt, 'w', encoding='utf-8') as f:
        for head, rel, tail in triples:
            f.write(f'{head}\t{rel}\t{tail}\n')
