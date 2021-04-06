import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle

import data.irt.split.split_dir
import data.power.split.split_dir
from data.irt.text.text_dir import TextDir
from data.power.split.facts_tsv import Fact


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    create_split(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('irt_split_dir', metavar='irt-split-dir',
                        help='Path to (input) IRT Split Directory')

    parser.add_argument('text_dir', metavar='text-dir',
                        help='Path to (input) IRT Text Directory')

    parser.add_argument('power_split_dir', metavar='power-split-dir',
                        help='Path to (output) POWER Split Directory')

    parser.add_argument('--known', dest='known', type=int, metavar='INT',
                        help='Percentage of known facts for test entities')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('irt-split-dir', args.irt_split_dir))
    logging.info('    {:24} {}'.format('text-dir', args.text_dir))
    logging.info('    {:24} {}'.format('power-split-dir', args.power_split_dir))
    logging.info('    {:24} {}'.format('--known', args.known))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def create_split(args):
    irt_split_dir_path = args.irt_split_dir
    text_dir_path = args.text_dir
    power_split_dir_path = args.power_split_dir

    known = args.known
    overwrite = args.overwrite

    #
    # Check that (input) IRT Split Directory exists
    #

    logging.info('Check that (input) IRT Split Directory exists ...')

    irt_split_dir = data.irt.split.split_dir.SplitDir(Path(irt_split_dir_path))
    irt_split_dir.check()

    #
    # Check that (input) IRT Text Directory exists
    #

    logging.info('Check that (input) IRT Text Directory exists ...')

    text_dir = TextDir(Path(text_dir_path))
    text_dir.check()

    #
    # Check that (output) POWER Split Directory does not exist
    #

    logging.info('Check that (output) POWER Split Directory does not exist ...')

    power_split_dir = data.power.split.split_dir.SplitDir(Path(power_split_dir_path))
    power_split_dir.create(overwrite=overwrite)

    #
    # Create POWER Entities/Relations TSVs
    #

    logging.info('Create POWER Entities/Relations TSVs ...')
    
    ent_to_lbl = irt_split_dir.ent_labels_txt.load()
    rel_to_lbl = irt_split_dir.rel_labels_txt.load()

    power_split_dir.entities_tsv.save(ent_to_lbl)
    power_split_dir.relations_tsv.save(rel_to_lbl)
    
    #
    # Create POWER Train/Valid/Test Entities TSVs
    #

    logging.info('Create POWER Train/Valid/Test Entities TSVs ...')

    train_texts = text_dir.cw_train_sents_txt.load()
    valid_texts = text_dir.ow_valid_sents_txt.load()
    test_texts = text_dir.ow_test_sents_txt.load()
    
    train_ent_to_lbl = {ent: lbl for ent, lbl in ent_to_lbl.items() if ent in train_texts}
    valid_ent_to_lbl = {ent: lbl for ent, lbl in ent_to_lbl.items() if ent in valid_texts}
    test_ent_to_lbl = {ent: lbl for ent, lbl in ent_to_lbl.items() if ent in test_texts}
    
    power_split_dir.train_entities_tsv.save(train_ent_to_lbl)
    power_split_dir.valid_entities_tsv.save(valid_ent_to_lbl)
    power_split_dir.test_entities_tsv.save(test_ent_to_lbl)

    #
    # Create POWER Train Facts TSV
    #

    logging.info('Create POWER Train Facts TSV ...')

    cw_train_triples = irt_split_dir.cw_train_triples_txt.load()
    cw_valid_triples = irt_split_dir.cw_valid_triples_txt.load()

    cw_triples = cw_train_triples + cw_valid_triples
    shuffle(cw_triples)

    train_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                   for head, rel, tail in cw_triples]

    power_split_dir.train_facts_tsv.save(train_facts)

    #
    # Create Valid Known/Unknown TSV
    #

    logging.info('Create Valid Known/Unknown TSV ...')

    ow_valid_triples = irt_split_dir.ow_valid_triples_txt.load()
    shuffle(ow_valid_triples)

    valid_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                   for head, rel, tail in ow_valid_triples]

    valid_facts_count = len(valid_facts)
    valid_known_count = int(valid_facts_count * known / 100)

    power_split_dir.valid_facts_known_tsv.save(valid_facts[:valid_known_count])
    power_split_dir.valid_facts_unknown_tsv.save(valid_facts[valid_known_count:])

    #
    # Create Neo4j Test Facts TSVs
    #

    logging.info('Create Neo4j Test Facts TSVs ...')

    ow_test_triples = irt_split_dir.ow_test_triples_txt.load()
    shuffle(ow_test_triples)

    test_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                  for head, rel, tail in ow_test_triples]

    test_facts_count = len(test_facts)
    test_known_count = int(test_facts_count * known / 100)

    power_split_dir.test_facts_known_tsv.save(test_facts[:test_known_count])
    power_split_dir.test_facts_unknown_tsv.save(test_facts[test_known_count:])


if __name__ == '__main__':
    main()
