from os import path
from os.path import isdir, isfile
from typing import Dict


def assert_existing(files: Dict, ryn_split_dir: str):
    """
    Assert that 'Ryn Split Directory' exists
    """

    files['ryn_split'] = {}

    if not isdir(ryn_split_dir):
        print("'Ryn Split Directory' not found")
        exit()

    cw_train_txt = path.join(ryn_split_dir, 'cw.train2id.txt')
    files['ryn_split']['cw_train_txt'] = cw_train_txt
    if not isfile(cw_train_txt):
        print("'Ryn Split Directory' / 'CW Train TXT' not found")
        exit()

    cw_valid_txt = path.join(ryn_split_dir, 'cw.valid2id.txt')
    files['ryn_split']['cw_valid_txt'] = cw_valid_txt
    if not isfile(cw_valid_txt):
        print("'Ryn Split Directory' / 'CW Valid TXT' not found")
        exit()

    relation_label_to_rid_txt = path.join(ryn_split_dir, 'relation2id.txt')
    files['ryn_split']['relation_label_to_rid_txt'] = relation_label_to_rid_txt
    if not isfile(relation_label_to_rid_txt):
        print("'Ryn Split Directory' / 'Relation Label-to-RID TXT' not found")
        exit()

    entity_label_to_rid_txt = path.join(ryn_split_dir, 'entity2id.txt')
    files['ryn_split']['entity_label_to_rid_txt'] = entity_label_to_rid_txt
    if not isfile(entity_label_to_rid_txt):
        print("'Ryn Split Directory' / 'Entity Label-to-RID TXT' not found")
        exit()
