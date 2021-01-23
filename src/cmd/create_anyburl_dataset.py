import random
from argparse import ArgumentParser
from os import makedirs, path, remove
from os.path import isdir, isfile
from typing import List, Dict

import dao.oid_to_rid_txt
import dao.anyburl.triples_txt
import dao.ryn.oid_to_rel_txt
import dao.ryn.triples_txt


def main() -> None:
    #
    # Parse args
    #

    parser = ArgumentParser(description="Create an 'AnyBURL Dataset' from a 'Ryn Split'")

    parser.add_argument('ryn_split_dir', metavar='ryn-split-dir',
                        help="Path to (input) 'Ryn Split Directory'")

    parser.add_argument('entity_oid_to_rid_txt', metavar='entity-oid-to-rid-txt',
                        help="Path to (input) 'Entity OID-to-RID TXT'")

    parser.add_argument('anyburl_dataset_dir', metavar='anyburl-dataset-dir',
                        help="Path to (output) 'AnyBURL Dataset Directory'")

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    args = parser.parse_args()

    ryn_split_dir = args.ryn_split_dir
    entity_oid_to_rid_txt = args.entity_oid_to_rid_txt
    anyburl_dataset_dir = args.anyburl_dataset_dir

    overwrite = args.overwrite

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:24} {}'.format('ryn-split-dir', ryn_split_dir))
    print('    {:24} {}'.format('entity-oid-to-rid-txt', entity_oid_to_rid_txt))
    print('    {:24} {}'.format('anyburl-dataset-dir', anyburl_dataset_dir))
    print()
    print('    {:24} {}'.format('--overwrite', overwrite))
    print()

    files = {}

    #
    # Assert that (input) 'Entity OID-to-RID TXT' exists
    #

    if not isfile(entity_oid_to_rid_txt):
        print("'Entity OID-to-RID TXT' not found")
        exit()

    files['entity-oid-to-rid-txt'] = entity_oid_to_rid_txt

    #
    # Assert that (input) 'Ryn Split Directory' exists
    #

    if not isdir(ryn_split_dir):
        print("'Ryn Split Directory' not found")
        exit()

    files['ryn-split'] = {}

    cw_train_txt = path.join(ryn_split_dir, 'cw.train2id.txt')
    if not isfile(cw_train_txt):
        print("'Ryn Split Directory' / 'CW Train TXT' not found")
        exit()

    files['ryn-split']['cw-train-txt'] = cw_train_txt

    cw_valid_txt = path.join(ryn_split_dir, 'cw.valid2id.txt')
    if not isfile(cw_valid_txt):
        print("'Ryn Split Directory' / 'CW Valid TXT' not found")
        exit()

    files['ryn-split']['cw-valid-txt'] = cw_valid_txt

    oid_to_rel_txt = path.join(ryn_split_dir, 'relation2id.txt')
    if not isfile(cw_train_txt):
        print("'Ryn Split Directory' / 'OID-to-Rel TXT' not found")
        exit()

    files['ryn-split']['oid-to-rel-txt'] = oid_to_rel_txt

    #
    # Check (output) 'AnyBURL Dataset Directory'
    # - Create it if it does not already exist
    # - Assert that its files do not already exist
    #

    makedirs(anyburl_dataset_dir, exist_ok=True)

    files['anyburl'] = {}

    train_txt = path.join(anyburl_dataset_dir, 'train.txt')
    if isfile(train_txt):
        if overwrite:
            remove(train_txt)
        else:
            print("'AnyBURL Dataset Directory' / 'Train TXT' already exists"
                  ", use --overwrite to overwrite it")
            exit()

    files['anyburl']['train-txt'] = train_txt

    makedirs(anyburl_dataset_dir, exist_ok=True)

    valid_txt = path.join(anyburl_dataset_dir, 'valid.txt')
    if isfile(valid_txt):
        if overwrite:
            remove(valid_txt)
        else:
            print("'AnyBURL Dataset Directory' / 'Valid TXT' already exists"
                  ", use --overwrite to overwrite it")
            exit()

    files['anyburl']['valid-txt'] = valid_txt

    test_txt = path.join(anyburl_dataset_dir, 'test.txt')
    if isfile(test_txt):
        if overwrite:
            remove(test_txt)
        else:
            print("'AnyBURL Dataset Directory' / 'Test TXT' already exists"
                  ", use --overwrite to overwrite it")
            exit()

    files['anyburl']['test-txt'] = test_txt

    #
    # Run actual program
    #

    create_anyburl_dataset(files)


def create_anyburl_dataset(files: Dict) -> None:
    #
    # Load triples from Ryn Split Dataset
    #

    print()
    print("Load RID triples from 'Ryn Split Dataset'...")

    train_valid_rid_triples_txt = files['ryn-split']['cw-train-txt']
    test_rid_triples_txt = files['ryn-split']['cw-valid-txt']

    train_valid_rid_triples: List[(int, int, int)] = dao.ryn.triples_txt.load(train_valid_rid_triples_txt)
    test_rid_triples: List[(int, int, int)] = dao.ryn.triples_txt.load(test_rid_triples_txt)

    print('Done')

    #
    # Split train/valid triples
    #

    random.shuffle(train_valid_rid_triples)

    split = int(0.7 * len(train_valid_rid_triples))

    train_rid_triples = train_valid_rid_triples[:split]
    valid_rid_triples = train_valid_rid_triples[split:]

    #
    # Load
    # - Entity RID -> OID mapping (from 'Entity OID-to-RID TXT')
    # - Relation RID -> OID mapping (from 'Ryn Split Dataset' / 'OID-to-Rel TXT')
    #

    print()
    print('Load RID -> OID mappings...')

    ent_oid_to_rid_txt = files['entity-oid-to-rid-txt']
    ent_oid_to_rid = dao.oid_to_rid_txt.load(ent_oid_to_rid_txt)

    ent_rid_to_oid = {rid: oid for oid, rid in ent_oid_to_rid.items()}

    rel_oid_to_rid_txt = files['ryn-split']['oid-to-rel-txt']
    rel_oid_to_rid: Dict[str, int] = dao.ryn.oid_to_rel_txt.load(rel_oid_to_rid_txt)

    rel_rid_to_oid = {rel: label for label, rel in rel_oid_to_rid.items()}

    print('Done')

    #
    # Save OID triples to 'AnyBURL Dataset Directory'
    #

    print()
    print("Save OID triples to 'AnyBURL Dataset Directory'")

    anyburl_train_txt = files['anyburl']['train-txt']
    anyburl_valid_txt = files['anyburl']['valid-txt']
    anyburl_test_txt = files['anyburl']['test-txt']

    train_oid_triples = [(ent_rid_to_oid[head], rel_rid_to_oid[rel], ent_rid_to_oid[tail])
                         for head, rel, tail in train_rid_triples]

    valid_oid_triples = [(ent_rid_to_oid[head], rel_rid_to_oid[rel], ent_rid_to_oid[tail])
                         for head, rel, tail in valid_rid_triples]

    test_oid_triples = [(ent_rid_to_oid[head], rel_rid_to_oid[rel], ent_rid_to_oid[tail])
                        for head, rel, tail in test_rid_triples]

    dao.anyburl.triples_txt.save(anyburl_train_txt, train_oid_triples)
    dao.anyburl.triples_txt.save(anyburl_valid_txt, valid_oid_triples)
    dao.anyburl.triples_txt.save(anyburl_test_txt, test_oid_triples)

    print('Done')


if __name__ == '__main__':
    main()
