from dataclasses import dataclass
from enum import Enum
from typing import Union, Dict, Optional

import data
from neo4j import Record

from models.var import Var
from models.ent import Ent
from models.rel import Rel


class Type(Enum):
    cw_train = 1
    cw_valid = 2
    ow_valid = 3
    ow_test = 4


@dataclass(frozen=True)
class Fact:
    head: Union[Ent, Var]
    rel: Rel
    tail: Union[Ent, Var]
    type: Optional[Type]

    @classmethod
    def from_anyburl(cls,
                     fact: data.anyburl.rules.rules_tsv.Fact,
                     ent_to_lbl: Dict[int, str],
                     rel_to_lbl: Dict[int, str]):
        head, rel, tail = fact

        return cls(Ent(head, ent_to_lbl[head]) if type(head) == int else Var(head),
                   Rel(rel, rel_to_lbl[rel]),
                   Ent(tail, ent_to_lbl[tail]) if type(tail) == int else Var(tail),
                   Type.cw_train)

    @classmethod
    def from_neo4j(cls, record: Record):
        return cls(Ent(record[0].get('id'), record[0].get('label')),
                   Rel(record[1].get('id'), record[1].get('label')),
                   Ent(record[2].get('id'), record[2].get('label')),
                   Type[record[1].get('type')])

    def __repr__(self):
        type_str = f'{self.type.name}: ' if self.type is not None else ''
        head_str = f'{self.head.lbl} ({self.head.id})' if type(self.head) == Ent else self.head.name
        tail_str = f'{self.tail.lbl} ({self.tail.id})' if type(self.tail) == Ent else self.tail.name

        return f'({type_str}{head_str} - {self.rel.lbl} ({self.rel.id}) -> {tail_str})'
