import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path

from data.anyburl.rules.rules_dir import RulesDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    train_ruler(args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('rules_dir', metavar='rules-dir',
                        help='Path to (input) AnyBURL Rules Directory')

    parser.add_argument('model_dir', metavar='model-dir',
                        help='Path to (output) POWER Model Directory')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('rules-dir', args.rules_dir))
    logging.info('    {:24} {}'.format('model-dir', args.model_dir))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def train_ruler(args):
    rules_dir_path = args.rules_dir
    model_dir_path = args.model_dir

    overwrite = args.overwrite

    #
    # Check (input) AnyBURL Rules Directory
    #

    logging.info('Check (input) AnyBURL Rules Directory ...')

    rules_dir = RulesDir(Path(rules_dir_path))
    rules_dir.check()

    #
    # Create (output) POWER Model Directory
    #

    logging.info('Create (output) POWER Model Directory ...')

    #
    # Read rules
    #

    rules = rules_dir.cw_train_rules_tsv.load()

    print(rules)


if __name__ == '__main__':
    main()
