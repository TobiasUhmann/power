"""
The `AnyBURL Rules TSV` contains the output rules mined by AnyBURL.

Format:
* Tabular separated values, no header
* 4 columns: Body count (int), head count (int), confidence (float), rule (str)

**Example**

::

    237  16  0.0675  member_of(Haiti,Y) <= member_of(A,Y)
    14   10  0.714   member_of(Haiti,Y) <= member_of(Uruguay,Y)
    12   9   0.75    member_of(Haiti,Y) <= member_of(Eritrea,Y)

|
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from data.base_file import BaseFile


@dataclass
class Fact:
    head: Union[int, str]
    rel: int
    tail: Union[int, str]


@dataclass
class Rule:
    fires: int
    holds: int
    confidence: float
    head: Fact
    body: List[Fact]

    def __iter__(self):
        return iter((self.fires, self.holds, self.confidence, self.head, self.body))


class RulesTsv(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def load(self) -> List[Rule]:
        with open(self.path) as f:
            lines = f.readlines()

        rules = []
        for line in lines:
            parts = line.split('\t')

            fires = int(parts[0])
            holds = int(parts[1])
            confidence = float(parts[2])

            rule_parts = parts[3].split(' <= ')

            rule_head = self.parse_fact(rule_parts[0])

            rule_body = []
            for body_part in rule_parts[1].split(', '):
                body_fact = self.parse_fact(body_part)
                rule_body.append(body_fact)

            rules.append(Rule(fires, holds, confidence, rule_head, rule_body))

        return rules

    @staticmethod
    def parse_fact(fact_str):
        match = re.match(r'(.*)\((.*),(.*)\)', fact_str)

        head_str = match.group(2)
        head = head_str if len(head_str) == 1 else int(head_str.split('_')[0])

        rel_str = match.group(1)
        rel = int(rel_str.split('_')[0])

        tail_str = match.group(3)
        tail = tail_str if len(tail_str) == 1 else int(tail_str.split('_')[0])

        return Fact(head, rel, tail)
