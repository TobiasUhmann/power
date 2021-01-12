from collections import defaultdict
from typing import Dict, Set


def read_contexts_txt(contexts_txt: str) -> Dict[int, Set[str]]:

    ent_to_contexts: Dict[int, Set[str]] = defaultdict(set)

    with open(contexts_txt, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        ent, _, context = line.split(' | ')
        ent_to_contexts[int(ent)].add(context.strip())

    return ent_to_contexts
