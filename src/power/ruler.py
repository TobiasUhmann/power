from typing import List, Dict, Tuple

from models.ent import Ent
from models.fact import Fact
from models.pred import Pred
from models.rel import Rel
from models.rule import Rule


class Ruler:
    pred: Dict[Ent, Dict[Tuple[Rel, Ent], List[Rule]]]

    def predict(self, ent: Ent) -> List[Pred]:
        preds = []

        for (rel, tail), rules in self.pred[ent].items():
            rules = list(rules)
            rules.sort(key=lambda rule: rule.conf, reverse=True)

            preds.append(Pred(Fact(ent, rel, tail), rules[0].conf, [], rules))

        return preds
