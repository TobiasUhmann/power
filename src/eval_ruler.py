import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from data.irt.split.split_dir import SplitDir
from data.power.model.model_dir import ModelDir
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

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) IRT Split Directory')

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
    # Check (input) IRT Split Directory
    #

    logging.info('Check (input) IRT Split Directory ...')

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

    skt_gt = []
    skt_pred = []

    for ent in tqdm(ents):
        # print()
        # print('ENT')
        # pprint(ent)

        gt = [fact for fact in cw_valid_facts if fact.head == ent]
        # print('GT')
        # pprint(gt)

        # pred = ruler[ent]
        # print('PRED')
        # pprint(pred)

        pred = [Fact(ent, rel, tail) for (rel, tail) in ruler[ent]]
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
