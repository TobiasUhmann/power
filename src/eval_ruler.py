import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from data.power.ruler_pkl import RulerPkl
from data.power.split.split_dir import SplitDir
from models.ent import Ent
from models.fact import Fact
from models.pred import Pred
from util import calc_ap
import numpy as np


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()
    log_config(args)

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

    return parser.parse_args()


def log_config(args):
    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ruler-pkl', args.ruler_pkl))
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('--filter-known', args.filter_known))
    logging.info('    {:24} {}'.format('--test', args.test))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))


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

    logging.info('Evaluate ...')

    all_gt_bools = []
    all_pred_bools = []

    all_prfs = []

    all_ap = []

    if logging.getLogger().level == logging.DEBUG:
        iter_eval_ents = eval_ents
    else:
        iter_eval_ents = tqdm(eval_ents)

    for ent in iter_eval_ents:
        logging.debug(f'Evaluate entity {ent} ...')

        #
        # Predict entity facts
        #

        preds: List[Pred] = ruler.predict(ent)

        if filter_known:
            preds = [pred for pred in preds if pred.fact not in known_facts]

        logging.debug('Predictions:')
        for pred in preds:
            logging.debug(str(pred))

        #
        # Get entity ground truth facts
        #

        gt_facts = [fact for fact in all_eval_facts if fact.head == ent]

        if filter_known:
            gt_facts = list(set(gt_facts).difference(known_facts))

        logging.debug('Ground truth:')
        for fact in gt_facts:
            logging.debug(str(fact))

        #
        # Calc entity PRFS
        #

        pred_facts = {pred.fact for pred in preds}
        pred_and_gt_facts = list(pred_facts | set(gt_facts))

        gt_bools = [1 if fact in gt_facts else 0 for fact in pred_and_gt_facts]
        pred_bools = [1 if fact in pred_facts else 0 for fact in pred_and_gt_facts]

        prfs = precision_recall_fscore_support(gt_bools, pred_bools, labels=[1], zero_division=1)
        all_prfs.append(prfs)

        #
        # Add ent results to global results for micro metrics
        #

        all_gt_bools.extend(gt_bools)
        all_pred_bools.extend(pred_bools)

        #
        # Calc entity AP
        #

        pred_fact_conf_tuples = [(pred.fact, pred.conf) for pred in preds]

        ap = calc_ap(pred_fact_conf_tuples, gt_facts)
        all_ap.append(ap)

        #
        # Log entity metrics
        #

        logging.debug(f'{str(ent.id):5} {ent.lbl:40}: AP = {ap:.2f}, Prec = {prfs[0][0]:.2f}, Rec = {prfs[1][0]:.2f}, '
                     f'F1 = {prfs[2][0]:.2f}, Supp = {prfs[3][0]}')

    m_ap = sum(all_ap) / len(all_ap)
    logging.info(f'mAP = {m_ap:.4f}')

    macro_prfs = np.array(all_prfs).mean(axis=0)
    logging.info(f'Macro Prec = {macro_prfs[0][0]:.4f}')
    logging.info(f'Macro Rec = {macro_prfs[1][0]:.4f}')
    logging.info(f'Macro F1 = {macro_prfs[2][0]:.4f}')
    logging.info(f'Macro Supp = {macro_prfs[3][0]:.2f}')

    micro_prfs = precision_recall_fscore_support(all_gt_bools, all_pred_bools, labels=[1], zero_division=1)
    logging.info(f'Micro Prec = {micro_prfs[0][0]:.4f}')
    logging.info(f'Micro Rec = {micro_prfs[1][0]:.4f}')
    logging.info(f'Micro F1 = {micro_prfs[2][0]:.4f}')
    logging.info(f'Micro Supp = {micro_prfs[3][0]}')


def get_defaultdict():
    return defaultdict(list)


if __name__ == '__main__':
    main()
