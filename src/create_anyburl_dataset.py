import logging
import os
import random
import re
from argparse import ArgumentParser
from pathlib import Path

from data.anyburl.facts_tsv import Fact, FactsTsv
from data.power.split.split_dir import SplitDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    create_anyburl_dataset(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) POWER Split Directory')

    parser.add_argument('facts_tsv', metavar='facts-tsv',
                        help='Path to (output) AnyBURL Facts TSV')

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
    logging.info('    {:24} {}'.format('facts-tsv', args.facts_tsv))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))
    logging.info('    {:24} {}'.format('--random-seed', args.random_seed))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def create_anyburl_dataset(args):
    split_dir_path = args.split_dir
    facts_tsv_path = args.facts_tsv

    overwrite = args.overwrite

    #
    # Check that (input) IRT Split Directory exists
    #

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Check that (output) AnyBURL Facts TSV does not exist
    #

    facts_tsv = FactsTsv(Path(facts_tsv_path))
    if not overwrite:
        facts_tsv.check(should_exist=False)

    facts_tsv.path.parent.mkdir(parents=True, exist_ok=True)

    #
    # Create AnyBURL Facts TSV
    #

    ent_to_lbl = split_dir.entities_tsv.load()
    rel_to_lbl = split_dir.relations_tsv.load()

    def escape(text):
        return re.sub('[^0-9a-zA-Z]', '_', text)

    def stringify_ent(ent):
        return f'{ent}_{escape(ent_to_lbl[ent])}'

    def stringify_rel(rel):
        return f'{rel}_{escape(rel_to_lbl[rel])}'

    train_facts = split_dir.train_facts_tsv.load()

    anyburl_facts = [Fact(stringify_ent(head), stringify_rel(rel), stringify_ent(tail))
                     for head, _, rel, _, tail, _ in train_facts]

    facts_tsv.save(anyburl_facts)


if __name__ == '__main__':
    main()
