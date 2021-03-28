import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle

from data.anyburl.facts.facts_dir import FactsDir
from data.anyburl.facts.facts_tsv import Fact
from data.ryn.split.split_dir import SplitDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    create_anyburl_dataset(args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) Ryn Split Directory')

    parser.add_argument('facts_dir', metavar='facts-dir',
                        help='Path to (output) AnyBURL Facts Directory')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('facts-dir', args.facts_dir))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def create_anyburl_dataset(args):
    split_dir_path = args.split_dir
    facts_dir_path = args.facts_dir

    overwrite = args.overwrite

    #
    # Check (input) Ryn Split Directory
    #

    logging.info('Check (input) Ryn Split Directory ...')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Create (output) AnyBURL Facts Directory
    #

    logging.info('Create (output) AnyBURL Facts Directory ...')

    facts_dir = FactsDir(Path(facts_dir_path))
    facts_dir.create(overwrite=overwrite)

    #
    # Create dataset
    #

    logging.info('Create dataset ...')

    # Load ent/rel labels
    ent_to_lbl = split_dir.ent_labels_txt.load()
    rel_to_lbl = split_dir.rel_labels_txt.load()

    def stringify_ent(ent):
        return f"{ent}_{ent_to_lbl[ent].replace(' ', '_')}"

    def stringify_rel(rel):
        return f"{rel}_{rel_to_lbl[rel].replace(' ', '_')}"

    # Create AnyBURL CW Train Facts TSV
    cw_train_triples = split_dir.cw_train_triples_txt.load()
    shuffle(cw_train_triples)
    cw_train_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                      for head, rel, tail in cw_train_triples]
    facts_dir.cw_train_facts_tsv.save(cw_train_facts)

    # Create AnyBURL CW Valid Facts TSV
    cw_valid_triples = split_dir.cw_valid_triples_txt.load()
    shuffle(cw_valid_triples)
    cw_valid_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                      for head, rel, tail in cw_valid_triples]
    facts_dir.cw_valid_facts_tsv.save(cw_valid_facts)

    # Create AnyBURL OW Valid Facts TSV
    ow_valid_triples = split_dir.ow_valid_triples_txt.load()
    shuffle(ow_valid_triples)
    ow_valid_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                      for head, rel, tail in ow_valid_triples]
    facts_dir.ow_valid_facts_tsv.save(ow_valid_facts)

    # Create AnyBURL OW Test Facts TSV
    ow_test_triples = split_dir.ow_test_triples_txt.load()
    shuffle(ow_test_triples)
    ow_test_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                     for head, rel, tail in ow_test_triples]
    facts_dir.ow_test_facts_tsv.save(ow_test_facts)

    logging.info('Finished successfully')


if __name__ == '__main__':
    main()
