"""
Functions for checking the file structure of a Ryn Split Directory
"""
from os import makedirs
from os.path import isdir
from pathlib import Path

from dao.ryn.split.labels_txt import LabelsTxt
from dao.ryn.split.triples_txt import TriplesTxt


class SplitDir:
    name: str
    path: Path

    entity_labels_txt: LabelsTxt
    relation_labels_txt: LabelsTxt
    
    cw_train_triples_txt: TriplesTxt
    cw_valid_triples_txt: TriplesTxt
    ow_valid_triples_txt: TriplesTxt
    ow_test_triples_txt: TriplesTxt

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

        self.entity_labels_txt = LabelsTxt('Entity Labels TXT', path.joinpath('entity2id.txt'))
        self.relation_labels_txt = LabelsTxt('Relation Labels TXT', path.joinpath('relation2id.txt'))
        
        self.cw_train_triples_txt = TriplesTxt('CW Train Triples TXT', path.joinpath('cw.train2id.txt'))
        self.cw_valid_triples_txt = TriplesTxt('CW Valid Triples TXT', path.joinpath('cw.valid2id.txt'))
        self.ow_valid_triples_txt = TriplesTxt('OW Valid Triples TXT', path.joinpath('ow.valid2id.txt'))
        self.ow_test_triples_txt = TriplesTxt('OW Test Triples TXT', path.joinpath('ow.test2id.txt'))

    def check(self) -> None:
        if not isdir(self.path):
            print(f'{self.name} not found')
            exit()
            
        self.entity_labels_txt.check()
        self.relation_labels_txt.check()

        self.cw_train_triples_txt.check()
        self.cw_valid_triples_txt.check()
        self.ow_valid_triples_txt.check()
        self.ow_test_triples_txt.check()

    def create(self) -> None:
        makedirs(self.path, exist_ok=True)
