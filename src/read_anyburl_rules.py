import re
from collections import Counter
from dataclasses import dataclass
from typing import List


@dataclass
class Fact:
    head: str
    rel: str
    tail: str

    def contains_constant(self):
        return len(self.head) > 1 or len(self.tail) > 1


@dataclass
class Rule:
    fires: int
    holds: int
    confidence: float
    head: Fact
    body: List[Fact]


if __name__ == '__main__':
    with open('../data/anyburl/rules/FB15-237/alpha-100') as fh:
        lines = fh.readlines()

    rules = []
    for line in lines:
        parts = line.split('\t')

        fires = int(parts[0])
        holds = int(parts[1])
        confidence = float(parts[2])

        rule_parts = parts[3].split(' <= ')

        head_match = re.match(r'(.*)\((.*),(.*)\)', rule_parts[0])
        head = Fact(head_match.group(2), head_match.group(1), head_match.group(3))

        body_parts = rule_parts[1].split(', ')
        body = []
        for body_part in body_parts:
            body_match = re.match(r'(.*)\((.*),(.*)\)', body_part)
            body.append(Fact(body_match.group(2), body_match.group(1), body_match.group(3)))

        rules.append(Rule(fires, holds, confidence, head, body))

    constant_rules = [rule for rule in rules if rule.head.contains_constant()]

    counter = Counter()
    for rule in constant_rules:
        counter[len(rule.body)] += 1

    long_rules = [rule for rule in rules if len(rule.body) > 1]

    good_rules = [rule for rule in constant_rules if rule.fires > 100 and rule.confidence > 0.5]
    good_rules.sort(key=lambda rule: rule.confidence, reverse=True)

    for good_rule in good_rules[:20]:
        print(good_rule)

    print(len(rules))
    print(len(good_rules))
    # print(counter)
    #
    # print(len(constant_rules))
    #
    # for constant_rule in constant_rules[:20]:
    #     print(constant_rule)

    #
    # Save rules in DB
    #


