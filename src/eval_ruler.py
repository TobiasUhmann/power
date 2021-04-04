import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from data.irt.text.text_dir import TextDir
from data.power.ruler.ruler_pkl import RulerPkl
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
    logging.info('    {:24} {}'.format('--test', args.test))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def eval_ruler(args):
    ruler_pkl_path = args.ruler_pkl
    split_dir_path = args.split_dir

    test = args.test

    #
    # Check that (input) POWER Ruler PKL exists
    #

    ruler_pkl = RulerPkl(Path(ruler_pkl_path))
    ruler_pkl.check()

    #
    # Check that (input) POWER Split Directory exists
    #

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

    ent_to_lbl = split_dir.entities_tsv.load()
    rel_to_lbl = split_dir.relations_tsv.load()

    if test:
        known_test_facts = split_dir.test_facts_known_tsv.load()
        unknown_test_facts = split_dir.test_facts_unknown_tsv.load()
        test_facts = known_test_facts + unknown_test_facts

        kfacts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                  for head, _, rel, _, tail, _ in known_test_facts}

    else:
        known_valid_facts = split_dir.valid_facts_known_tsv.load()
        unknown_valid_facts = split_dir.valid_facts_unknown_tsv.load()
        test_facts = known_valid_facts + unknown_valid_facts

        kfacts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                  for head, _, rel, _, tail, _ in known_valid_facts}

    test_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                  for head, _, rel, _, tail, _ in test_facts}

    #
    #
    #

    text_dir = TextDir(Path('data/irt/cde/text/cde/bert-base-cased.1.768.clean'))
    v_ents = text_dir.ow_test_sents_txt.load()
    valid_ents = [Ent(id, ent_to_lbl[id]) for id in v_ents]

    skt_gt = []
    skt_pred = []

    for ent in tqdm(valid_ents):
        # print()
        # print('ENT')
        # pprint(ent)

        gt = list(set(test_facts).difference(set(kfacts)))
        gt = [fact for fact in gt if fact.head == ent]
        # print('GT')
        # pprint(gt)

        # pred = ruler[ent]
        # print('PRED')
        # pprint(pred)

        pred = [Fact(ent, rel, tail) for (rel, tail) in ruler.pred[ent]]
        pred = list(set(pred).difference(set(kfacts)))
        # print('PRED')
        # pprint(pred)

        union = list(set(gt) | set(pred))
        # print('UNION')
        # pprint(union)

        sk_gt = [1 if fact in gt else 0 for fact in union]
        sk_pred = [1 if fact in pred else 0 for fact in union]

        skt_gt.extend(sk_gt)
        skt_pred.extend(sk_pred)

        prfs = precision_recall_fscore_support(sk_gt, sk_pred, labels=[1], zero_division=0)
        # print('PRFS')
        # print(sk_gt)
        # print(sk_pred)
        # print(prfs)

        pass

    prfs = precision_recall_fscore_support(skt_gt, skt_pred, labels=[1])
    print('PRFS')
    print(skt_gt)
    print(skt_pred)
    print(prfs)


def get_defaultdict():
    return defaultdict(list)


if __name__ == '__main__':
    main()
