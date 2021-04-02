from typing import List, Tuple, Dict

from models.ent import Ent
from models.fact import Fact
from models.pred import Pred
from models.rule import Rule
from power.ruler import Ruler
from power.texter import Texter


class Aggregator:
    texter: Texter
    ruler: Ruler

    def __init__(self, texter: Texter, ruler: Ruler):
        super().__init__()

        self.texter = texter
        self.ruler = ruler

    def predict(self, ent: Ent, sents: List[str]) -> List[Pred]:
        preds = []

        text_preds: Dict[Fact, List[Tuple[str, float]]] = self.texter.predict(ent, sents)
        rule_preds: Dict[Fact, List[Rule]] = self.ruler.predict(ent)

        for fact in text_preds.keys() | rule_preds.keys():
            sents = text_preds[fact] if fact in text_preds else []
            rules = rule_preds[fact] if fact in rule_preds else []

            max_sent_conf = sents[0][1] if len(sents) > 0 else 0
            max_rule_conf = rules[0].conf if len(rules) > 0 else 0

            conf = max(max_sent_conf, max_rule_conf)

            preds.append(Pred(fact, conf, sents, rules))

        preds.sort(key=lambda pred: pred.conf, reverse=True)

        return preds
