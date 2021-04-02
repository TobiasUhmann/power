from typing import List, Dict, Tuple

from models.ent import Ent
from models.fact import Fact
from models.rel import Rel
from models.rule import Rule


class Ruler:
    pred: Dict[Ent, Dict[Tuple[Rel, Ent], List[Rule]]]

    def predict(self, ent: Ent) -> Dict[Fact, List[Rule]]:
        return {Fact(ent, rel, tail): rules
                for (rel, tail), rules in self.pred[ent].items()}
