"""
Functions for checking the file structure of a Ryn Split Directory
"""

from os.path import isdir
from pathlib import Path

from dao.ryn.split.cw_train_triples_txt import CwTrainTriplesTxt
from dao.ryn.split.cw_valid_triples_txt import CwValidTriplesTxt
from dao.ryn.split.entity_labels_txt import EntityLabelsTxt
from dao.ryn.split.ow_test_triples_txt import OwTestTriplesTxt
from dao.ryn.split.ow_valid_triples_txt import OwValidTriplesTxt
from dao.ryn.split.relation_labels_txt import RelationLabelsTxt


class SplitDir:
    path: Path

    entity_labels_txt: EntityLabelsTxt
    relation_labels_txt: RelationLabelsTxt
    
    cw_train_triples_txt: CwTrainTriplesTxt
    cw_valid_triples_txt: CwValidTriplesTxt
    ow_valid_triples_txt: OwValidTriplesTxt
    ow_test_triples_txt: OwTestTriplesTxt

    def __init__(self, path: Path):
        self.path = path

        self.entity_labels_txt = EntityLabelsTxt(path.joinpath('entity2id.txt'))
        self.relation_labels_txt = RelationLabelsTxt(path.joinpath('relation2id.txt'))
        
        self.cw_train_triples_txt = CwTrainTriplesTxt(path.joinpath('cw.train2id.txt'))
        self.cw_valid_triples_txt = CwValidTriplesTxt(path.joinpath('cw.valid2id.txt'))
        self.ow_valid_triples_txt = OwValidTriplesTxt(path.joinpath('ow.valid2id.txt'))
        self.ow_test_triples_txt = OwTestTriplesTxt(path.joinpath('ow.test2id.txt'))

    def check(self) -> None:
        if not isdir(self.path):
            print('Split Directory not found')
            exit()
            
        self.entity_labels_txt.check()
        self.relation_labels_txt.check()

        self.cw_train_triples_txt.check()
        self.cw_valid_triples_txt.check()
        self.ow_valid_triples_txt.check()
        self.ow_test_triples_txt.check()
