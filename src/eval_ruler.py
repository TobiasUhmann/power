import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

from data.power.ruler_pkl import RulerPkl
from data.power.split.split_dir import SplitDir
from models.ent import Ent
from models.fact import Fact


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    eval_ruler(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('ruler_pkl', metavar='ruler-pkl',
                        help='Path to (input) POWER Ruler PKL')

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) POWER Split Directory')

    parser.add_argument('--filter-known', dest='filter_known', action='store_true',
                        help='Filter out known valid triples')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    parser.add_argument('--test', dest='test', action='store_true',
                        help='Load test data instead of valid data')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ruler-pkl', args.ruler_pkl))
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('--filter-known', args.filter_known))
    logging.info('    {:24} {}'.format('--test', args.test))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def eval_ruler(args):
    ruler_pkl_path = args.ruler_pkl
    split_dir_path = args.split_dir

    filter_known = args.filter_known
    test = args.test

    #
    # Check that (input) POWER Ruler PKL exists
    #

    logging.info('Check that (input) POWER Ruler PKL exists ...')

    ruler_pkl = RulerPkl(Path(ruler_pkl_path))
    ruler_pkl.check()

    #
    # Check that (input) POWER Split Directory exists
    #

    logging.info('Check that (input) POWER Split Directory exists ...')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Load ruler
    #

    logging.info('Load ruler ...')

    ruler = ruler_pkl.load()

    #
    # Load facts
    #

    logging.info('Load facts ...')

    ent_to_lbl = split_dir.entities_tsv.load()
    rel_to_lbl = split_dir.relations_tsv.load()

    if test:
        known_test_facts = split_dir.test_facts_known_tsv.load()
        unknown_test_facts = split_dir.test_facts_unknown_tsv.load()

        known_eval_facts = known_test_facts
        all_eval_facts = known_test_facts + unknown_test_facts

    else:
        known_valid_facts = split_dir.valid_facts_known_tsv.load()
        unknown_valid_facts = split_dir.valid_facts_unknown_tsv.load()

        known_eval_facts = known_valid_facts
        all_eval_facts = known_valid_facts + unknown_valid_facts

    known_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                   for head, _, rel, _, tail, _ in known_eval_facts}

    all_eval_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                      for head, _, rel, _, tail, _ in all_eval_facts}

    #
    # Load entities
    #

    logging.info('Load entities ...')

    if test:
        eval_ents = split_dir.test_entities_tsv.load()
    else:
        eval_ents = split_dir.valid_entities_tsv.load()

    eval_ents = [Ent(ent, lbl) for ent, lbl in eval_ents.items()]

    #
    # Evaluate
    #

    total_gt = []
    total_pred = []

    for ent in eval_ents:
        logging.info(f'Evaluate entity "{ent.lbl}" ({ent.id}) ...')

        #
        # Get entity's ground truth facts
        #

        gt = [fact for fact in all_eval_facts if fact.head == ent]

        if filter_known:
            gt = list(set(gt).difference(known_facts))

        if filter_known:
            logging.debug('Ground truth (filtered):')
        else:
            logging.debug('Ground truth:')

        for fact in gt:
            logging.debug(str(fact))

        #
        # Get entity's ground truth facts
        #

        pred = [Fact(ent, rel, tail) for (rel, tail) in ruler.pred[ent]]

        if filter_known:
            pred = list(set(pred).difference(known_facts))

        if filter_known:
            logging.debug('Predicted (filtered):')
        else:
            logging.debug('Predicted:')

        for fact in gt:
            logging.debug(str(fact))

        #
        # Add ground truth and predicted facts to global sklearn lists
        #

        union = list(set(gt) | set(pred))

        union_gt = [1 if fact in gt else 0 for fact in union]
        union_pred = [1 if fact in pred else 0 for fact in union]

        total_gt.extend(union_gt)
        total_pred.extend(union_pred)

        ent_prfs = precision_recall_fscore_support(union_gt, union_pred, labels=[1], zero_division=0)
        logging.debug(f'PRFS: {ent_prfs}')

    total_prfs = precision_recall_fscore_support(total_gt, total_pred, labels=[1])
    logging.info(f'PRFS: {total_prfs}')


def get_defaultdict():
    return defaultdict(list)


if __name__ == '__main__':
    main()
