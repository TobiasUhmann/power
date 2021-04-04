from dataclasses import dataclass
from typing import Union, Dict

from neo4j import Record

import data.power.ruler.rules_tsv
from models.ent import Ent
from models.rel import Rel
from models.var import Var


@dataclass(frozen=True)
class Fact:
    head: Union[Ent, Var]
    rel: Rel
    tail: Union[Ent, Var]

    @staticmethod
    def from_ints(head: int, rel: int, tail: int, ent_to_lbl: Dict[int, str], rel_to_lbl: Dict[int, str]):
        return Fact(Ent(head, ent_to_lbl[head]),
                    Rel(rel, rel_to_lbl[rel]),
                    Ent(tail, ent_to_lbl[tail]))

    @staticmethod
    def from_anyburl(fact: data.power.ruler.rules_tsv.Fact,
                     ent_to_lbl: Dict[int, str],
                     rel_to_lbl: Dict[int, str]):
        head, rel, tail = fact

        return Fact(Ent(head, ent_to_lbl[head]) if type(head) == int else Var(head),
                    Rel(rel, rel_to_lbl[rel]),
                    Ent(tail, ent_to_lbl[tail]) if type(tail) == int else Var(tail))

    @staticmethod
    def from_neo4j(record: Record):
        return Fact(Ent(record[0].get('id'), record[0].get('label')),
                    Rel(record[1].get('id'), record[1].get('label')),
                    Ent(record[2].get('id'), record[2].get('label')))

    def __repr__(self):
        head_str = f'{self.head.lbl} ({self.head.id})' if type(self.head) == Ent else self.head.name
        tail_str = f'{self.tail.lbl} ({self.tail.id})' if type(self.tail) == Ent else self.tail.name

        return f'({head_str} - {self.rel.lbl} ({self.rel.id}) -> {tail_str})'
