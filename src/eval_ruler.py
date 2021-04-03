import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from data.irt.text.text_dir import TextDir
from data.power.model.ruler_pkl import RulerPkl
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

    parser.add_argument('train_facts_tsv', metavar='train-facts-tsv',
                        help='Path to (input) POWER Train Facts TSV')

    parser.add_argument('test_facts_tsvs', metavar='test-facts-tsvs', nargs='*',
                        help='Paths to (input) POWER Test Facts TSVs')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ruler-pkl', args.ruler_pkl))
    logging.info('    {:24} {}'.format('train-facts-tsv', args.train_facts_tsv))
    logging.info('    {:24} {}'.format('test-facts-tsvs', args.test_facts_tsvs))
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def eval_ruler(args):
    ruler_pkl_path = args.ruler_pkl

    train_facts_tsv_path = args.train_facts_tsv
    test_facts_tsv_paths = args.test_facts_tsv_paths

    #
    # Check that (input) POWER Ruler PKL exists
    #

    ruler_pkl = RulerPkl(Path(ruler_pkl_path))
    ruler_pkl.check()

    #
    # Load train/valid facts
    #

    ent_to_lbl = split_dir.ent_labels_tsv.load()
    rel_to_lbl = split_dir.rel_labels_tsv.load()

    valid_rows = split_dir.test_facts_25_1_tsv.load() + \
                 split_dir.test_facts_25_2_tsv.load() + \
                 split_dir.test_facts_25_3_tsv.load() + \
                 split_dir.test_facts_25_4_tsv.load()

    valid_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                   for head, _, rel, _, tail, _ in valid_rows}

    text_dir = TextDir(Path('data/irt/cde/text/cde/bert-base-cased.1.768.clean'))
    v_ents = text_dir.ow_test_sents_txt.load()

    #
    # Load ruler
    #

    logging.info('Load ruler ...')

    ruler = ruler_pkl.load()

    valid_ents = [Ent(id, ent_to_lbl[id]) for id in v_ents]

    skt_gt = []
    skt_pred = []

    for ent in tqdm(valid_ents):
        # print()
        # print('ENT')
        # pprint(ent)

        gt = [fact for fact in valid_facts if fact.head == ent]
        # print('GT')
        # pprint(gt)

        # pred = ruler[ent]
        # print('PRED')
        # pprint(pred)

        pred = [Fact(ent, rel, tail) for (rel, tail) in ruler.pred[ent]]
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
