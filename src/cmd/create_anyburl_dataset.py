import random
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict

from dao.anyburl.anyburl_dir import assert_not_existing
from dao.anyburl.anyburl_triples_txt import save_triples
from dao.ryn.split.split_dir import SplitDir


def main() -> None:
    #
    # Parse args
    #

    parser = ArgumentParser(description="Create an 'AnyBURL Dataset' from a 'Ryn Split'")

    parser.add_argument('ryn_split_dir', metavar='ryn-split-dir',
                        help="Path to (input) 'Ryn Split Directory'")

    parser.add_argument('anyburl_dataset_dir', metavar='anyburl-dataset-dir',
                        help="Path to (output) 'AnyBURL Dataset Directory'")

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    args = parser.parse_args()

    ryn_split_dir = args.ryn_split_dir
    anyburl_dataset_dir = args.anyburl_dataset_dir

    overwrite = args.overwrite

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:24} {}'.format('ryn-split-dir', ryn_split_dir))
    print('    {:24} {}'.format('anyburl-dataset-dir', anyburl_dataset_dir))
    print()
    print('    {:24} {}'.format('--overwrite', overwrite))
    print()

    #
    # Check files
    #

    split_dir = SplitDir('Ryn Directory', Path(ryn_split_dir))
    split_dir.check()

    assert_not_existing(anyburl_dataset_dir, files, overwrite)

    #
    # Run actual program
    #

    create_anyburl_dataset(files)


def create_anyburl_dataset(files: Dict) -> None:
    #
    # Load triples from Ryn Split Dataset
    #

    print()
    print("Load RID triples from 'Ryn Split Dataset' ...")

    train_valid_rid_triples_txt = files['ryn_split']['cw_train_txt']
    test_rid_triples_txt = files['ryn_split']['cw_valid_txt']

    train_valid_rid_triples: List[(int, int, int)] = load_triples(train_valid_rid_triples_txt)
    test_rid_triples: List[(int, int, int)] = load_triples(test_rid_triples_txt)

    print('Done')

    #
    # Split train/valid triples
    #

    random.shuffle(train_valid_rid_triples)

    split = int(0.7 * len(train_valid_rid_triples))

    train_rid_triples = train_valid_rid_triples[:split]
    valid_rid_triples = train_valid_rid_triples[split:]

    #
    # Load RID -> Label mappings (from 'Ryn Split Dataset')
    #

    print()
    print('Load RID -> Label mappings ...')

    entity_label_to_rid_txt = files['ryn_split']['entity_label_to_rid_txt']
    relation_label_to_rid_txt = files['ryn_split']['relation_label_to_rid_txt']

    entity_rid_to_label: Dict[int, str] = load_rid_to_label(entity_label_to_rid_txt)
    relation_rid_to_label: Dict[int, str] = load_rid_to_label(relation_label_to_rid_txt)

    print('Done')

    #
    # RID triples -> label triples
    #

    def ent_lbl(ent: int) -> str:
        return str(ent) + '_' + entity_rid_to_label[ent].replace(' ', '_')

    def rel_lbl(rel: int) -> str:
        return str(rel) + '_' + relation_rid_to_label[rel].replace(' ', '_')

    train_lbl_triples = [(ent_lbl(head), rel_lbl(rel), ent_lbl(tail))
                         for head, rel, tail in train_rid_triples]

    valid_lbl_triples = [(ent_lbl(head), rel_lbl(rel), ent_lbl(tail))
                         for head, rel, tail in valid_rid_triples]

    test_lbl_triples = [(ent_lbl(head), rel_lbl(rel), ent_lbl(tail))
                        for head, rel, tail in test_rid_triples]

    #
    # Save label triples to 'AnyBURL Dataset Directory'
    #

    print()
    print("Save label triples to 'AnyBURL Dataset Directory' ...")

    anyburl_train_txt = files['anyburl']['train_txt']
    anyburl_valid_txt = files['anyburl']['valid_txt']
    anyburl_test_txt = files['anyburl']['test_txt']

    save_triples(anyburl_train_txt, train_lbl_triples)
    save_triples(anyburl_valid_txt, valid_lbl_triples)
    save_triples(anyburl_test_txt, test_lbl_triples)

    print('Done')


if __name__ == '__main__':
    main()
