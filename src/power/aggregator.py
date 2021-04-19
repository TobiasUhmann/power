from random import shuffle
from typing import List

import torch
from torch import Tensor

from models.ent import Ent
from models.pred import Pred
from power.ruler import Ruler
from power.texter import Texter


class Aggregator:
    texter: Texter
    ruler: Ruler

    texter_weight: Tensor
    ruler_weight: Tensor

    def __init__(self, texter: Texter, ruler: Ruler):
        super().__init__()

        self.texter = texter
        self.ruler = ruler

        self.texter_weight = torch.tensor([1.], requires_grad=True)
        self.ruler_weight = torch.tensor([100000.], requires_grad=True)

    def predict(self, ent: Ent, sents: List[str]) -> List[Pred]:
        preds = []

        texter_preds = self.texter.predict(ent, sents)
        ruler_preds = self.ruler.predict(ent)

        texter_fact_to_pred = {pred.fact: pred for pred in texter_preds}
        ruler_fact_to_pred = {pred.fact: pred for pred in ruler_preds}

        pred_facts = texter_fact_to_pred.keys() | ruler_fact_to_pred.keys()

        for fact in pred_facts:
            sents = texter_fact_to_pred[fact].sents if fact in texter_fact_to_pred else []
            rules = ruler_fact_to_pred[fact].rules if fact in ruler_fact_to_pred else []

            max_sent_conf = texter_fact_to_pred[fact].conf * self.texter_weight.item() \
                if fact in texter_fact_to_pred else 0

            max_rule_conf = ruler_fact_to_pred[fact].conf * self.ruler_weight.item() \
                if fact in ruler_fact_to_pred else 0

            conf = max_sent_conf + max_rule_conf

            preds.append(Pred(fact, conf, sents, rules))

        preds.sort(key=lambda pred: pred.conf, reverse=True)

        return preds
