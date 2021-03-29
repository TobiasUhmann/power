import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from data.power.model.model_dir import ModelDir
from data.ryn.split.split_dir import SplitDir
from models.ent import Ent
from models.fact import Fact


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    eval_ruler(args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) Ryn Split Directory')

    parser.add_argument('model_dir', metavar='model-dir',
                        help='Path to (input) POWER Model Directory')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('model-dir', args.model_dir))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def eval_ruler(args):
    split_dir_path = args.split_dir
    model_dir_path = args.model_dir

    #
    # Check (input) Ryn Split Directory
    #

    logging.info('Check (input) Ryn Split Directory ...')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Check (input) POWER Model Directory
    #

    logging.info('Check (input) POWER Model Directory ...')

    model_dir = ModelDir(Path(model_dir_path))
    model_dir.check()

    #
    #
    #

    ent_to_lbl = split_dir.ent_labels_txt.load()
    rel_to_lbl = split_dir.rel_labels_txt.load()

    cw_train_triples = split_dir.cw_train_triples_txt.load()
    cw_train_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                      for head, rel, tail in cw_train_triples}

    cw_valid_triples = split_dir.cw_valid_triples_txt.load()
    cw_valid_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                      for head, rel, tail in cw_valid_triples}

    #
    # Load ruler
    #

    logging.info('Load ruler ...')

    ruler = model_dir.ruler_pkl.load()

    ents = [Ent(id, lbl) for id, lbl in ent_to_lbl.items()]
    print(ents)

    logging.info('Finished successfully')


def get_defaultdict():
    return defaultdict(list)


if __name__ == '__main__':
    main()
