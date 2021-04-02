from dataclasses import dataclass
from typing import List, Dict

import data.power.rules.rules_tsv

from models.fact import Fact


@dataclass(frozen=True)
class Rule:
    fires: int
    holds: int
    conf: float
    head: Fact
    body: List[Fact]

    @classmethod
    def from_anyburl(cls,
                     rule: data.power.rules.rules_tsv.Rule,
                     ent_to_lbl: Dict[int, str],
                     rel_to_lbl: Dict[int, str]):
        fires, holds, confidence, head, body = rule

        return Rule(fires,
                    holds,
                    confidence,
                    Fact.from_anyburl(head, ent_to_lbl, rel_to_lbl),
                    [Fact.from_anyburl(fact, ent_to_lbl, rel_to_lbl) for fact in body])

    def __repr__(self):
        return f"[fires={self.fires}, holds={self.holds}, conf={self.conf:.2f}," \
               f" {self.head} <= {', '.join([str(fact) for fact in self.body])}]"
