"""
The `Ryn Sentences TXT` stores the entities' sentences.

* Header row
* Entity RID | Entity label | Sentences

**Example**

::

    # Format: <ID> | <NAME> | <SENTENCE>
    0 | Dominican Republic | Border disputes under Trujillo ...
    0 | Dominican Republic | In 2006, processed shells were ...

|
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

from dao.base_file import BaseFile


class RynSentencesTxt(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    def load_ent_to_sentences(self) -> Dict[int, Set[str]]:
        ent_to_contexts: Dict[int, Set[str]] = defaultdict(set)

        with open(self._path, encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]:
            ent, _, context = line.split(' | ')
            ent_to_contexts[int(ent)].add(context.strip())

        return ent_to_contexts
