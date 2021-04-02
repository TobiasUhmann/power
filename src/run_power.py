import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from data.power.model.model_dir import ModelDir
from models.ent import Ent
from power.aggregator import Aggregator


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    run_power(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('model_dir', metavar='model-dir',
                        help='Path to (input) POWER Model Directory')

    parser.add_argument('entity', metavar='entity', type=int,
                        help='Entity ID')

    parser.add_argument('sentences', metavar='sentences', nargs='*',
                        help='Sentences describing the entity')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('model-dir', args.model_dir))
    logging.info('    {:24} {}'.format('entity', args.entity))
    logging.info('    {:24} {}'.format('sentences', args.sentences))
    logging.info('    {:24} {}'.format('--random-seed', args.random_seed))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def run_power(args):
    model_dir_path = args.model_dir
    entity = args.entity
    sentences = args.sentences

    #
    # Check (input) POWER Model Directory
    #

    logging.info('Check (input) POWER Model Directory ...')

    model_dir = ModelDir(Path(model_dir_path))
    model_dir.check()

    #
    # Load POWER
    #

    logging.info('Load POWER ...')

    texter = model_dir.texter_pkl.load().cpu()
    ruler = model_dir.ruler_pkl.load()

    power = Aggregator(texter, ruler)

    #
    # Run POWER
    #

    ent_to_lbl = model_dir.ent_labels_txt.load()

    preds = power.predict(Ent(entity, ent_to_lbl[entity]), sentences)
    pprint(preds)


def get_defaultdict():
    return defaultdict(list)


if __name__ == '__main__':
    main()
