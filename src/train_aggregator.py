import logging
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from data.irt.text.text_dir import TextDir
from data.power.ruler_pkl import RulerPkl
from data.power.split.split_dir import SplitDir
from data.power.texter_pkl import TexterPkl
from models.ent import Ent
from models.fact import Fact
from power.aggregator import Aggregator


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    train_texter(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('ruler_pkl', metavar='ruler-pkl',
                        help='Path to (input) POWER Ruler PKL')

    parser.add_argument('texter_pkl', metavar='texter-pkl',
                        help='Path to (input) POWER Texter PKL')

    parser.add_argument('sent_count', metavar='sent-count', type=int,
                        help='Number of sentences per entity')

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) POWER Split Directory')

    parser.add_argument('text_dir', metavar='text-dir',
                        help='Path to (input) IRT Text Directory')

    default_epoch_count = 20
    parser.add_argument('--epoch-count', dest='epoch_count', type=int, metavar='INT', default=default_epoch_count,
                        help='Number of training epochs (default: {})'.format(default_epoch_count))

    default_log_dir = None
    parser.add_argument('--log-dir', dest='log_dir', metavar='STR', default=default_log_dir,
                        help='Tensorboard log directory (default: {})'.format(default_log_dir))

    parser.add_argument('--log-steps', dest='log_steps', action='store_true',
                        help='Log after steps, otherwise log after epochs')

    default_learning_rate = 1e-5
    parser.add_argument('--lr', dest='lr', type=float, metavar='FLOAT', default=default_learning_rate,
                        help='Learning rate (default: {})'.format(default_learning_rate))

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    default_sent_len = 64
    parser.add_argument('--sent-len', dest='sent_len', type=int, metavar='INT', default=default_sent_len,
                        help='Sentence length short sentences are padded and long sentences cropped to'
                             ' (default: {})'.format(default_sent_len))

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ruler-pkl', args.ruler_pkl))
    logging.info('    {:24} {}'.format('texter-pkl', args.texter_pkl))
    logging.info('    {:24} {}'.format('sent-count', args.sent_count))
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('text-dir', args.text_dir))
    logging.info('    {:24} {}'.format('--epoch-count', args.epoch_count))
    logging.info('    {:24} {}'.format('--log-dir', args.log_dir))
    logging.info('    {:24} {}'.format('--log-steps', args.log_steps))
    logging.info('    {:24} {}'.format('--lr', args.lr))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))
    logging.info('    {:24} {}'.format('--random-seed', args.random_seed))
    logging.info('    {:24} {}'.format('--sent-len', args.sent_len))

    return args


def train_texter(args):
    ruler_pkl_path = args.ruler_pkl
    texter_pkl_path = args.texter_pkl
    sent_count = args.sent_count
    split_dir_path = args.split_dir
    text_dir_path = args.text_dir

    epoch_count = args.epoch_count
    log_dir = args.log_dir
    log_steps = args.log_steps
    lr = args.lr
    overwrite = args.overwrite
    sent_len = args.sent_len

    #
    # Check that (input) POWER Ruler PKL exists
    #

    logging.info('Check that (input) POWER Ruler PKL exists ...')

    ruler_pkl = RulerPkl(Path(ruler_pkl_path))
    ruler_pkl.check()

    #
    # Check that (input) POWER Texter PKL exists
    #

    logging.info('Check that (input) POWER Texter PKL exists ...')

    texter_pkl = TexterPkl(Path(texter_pkl_path))
    texter_pkl.check()

    #
    # Check that (input) POWER Split Directory exists
    #

    logging.info('Check that (input) POWER Split Directory exists ...')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Check that (input) IRT Text Directory exists
    #

    logging.info('Check that (input) IRT Text Directory exists ...')

    text_dir = TextDir(Path(text_dir_path))
    text_dir.check()

    #
    # Load ruler
    #

    logging.info('Load ruler ...')

    ruler = ruler_pkl.load()

    #
    # Load texter
    #

    logging.info('Load texter ...')

    texter = texter_pkl.load().cpu()

    #
    # Build POWER
    #

    power = Aggregator(texter, ruler)

    #
    # Load facts
    #

    logging.info('Load facts ...')

    ent_to_lbl = split_dir.entities_tsv.load()
    rel_to_lbl = split_dir.relations_tsv.load()

    train_facts = split_dir.train_facts_tsv.load()
    train_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                   for head, _, rel, _, tail, _ in train_facts}

    known_valid_facts = split_dir.valid_facts_known_tsv.load()
    unknown_valid_facts = split_dir.valid_facts_unknown_tsv.load()

    known_eval_facts = known_valid_facts
    all_valid_facts = known_valid_facts + unknown_valid_facts

    known_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                   for head, _, rel, _, tail, _ in known_eval_facts}

    all_valid_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                      for head, _, rel, _, tail, _ in all_valid_facts}

    #
    # Load entities
    #

    logging.info('Load entities ...')

    train_ents = split_dir.train_entities_tsv.load()
    valid_ents = split_dir.valid_entities_tsv.load()

    train_ents = [Ent(ent, lbl) for ent, lbl in train_ents.items()]
    valid_ents = [Ent(ent, lbl) for ent, lbl in valid_ents.items()]

    #
    # Load texts
    #

    logging.info('Load texts ...')

    train_ent_to_sents = text_dir.cw_train_sents_txt.load()
    valid_ent_to_sents = text_dir.ow_valid_sents_txt.load()

    #
    # Prepare training
    #

    criterion = MSELoss()

    writer = SummaryWriter(log_dir=log_dir)

    #
    # Train
    #

    logging.info('Train ...')

    texter_optimizer = SGD([power.texter_weight], lr=lr)
    ruler_optimizer = SGD([power.ruler_weight], lr=lr)

    for epoch in range(epoch_count):

        for ent in train_ents:
            print(power.texter_weight)
            print(power.ruler_weight)
            print()

            #
            # Get entity ground truth facts
            #

            gt_facts = [fact for fact in train_facts if fact.head == ent]

            logging.debug('Ground truth:')
            for fact in gt_facts:
                logging.debug(str(fact))

            #
            # Train Texter Weight
            #

            sents = list(train_ent_to_sents[ent.id])[:sent_count]
            if len(sents) < sent_count:
                logging.warning(f'Only {len(sents)} sentences for entity "{ent.lbl}" ({ent.id}). Skipping.')
                continue

            texter_preds = texter.predict(ent, sents)

            train_confs = [pred.conf for pred in texter_preds]
            gt_confs = [1 if pred.fact in gt_facts else 0 for pred in texter_preds]

            for train_conf, gt_conf in zip(train_confs, gt_confs):
                loss = criterion(torch.tensor(train_conf) * power.texter_weight, torch.tensor(gt_conf).float())
                texter_optimizer.zero_grad()
                loss.backward()
                texter_optimizer.step()

            #
            # Train Ruler Weight
            #

            ruler_preds = ruler.predict(ent)

            train_confs = [pred.conf for pred in ruler_preds]
            gt_confs = [1 if pred.fact in gt_facts else 0 for pred in ruler_preds]

            for train_conf, gt_conf in zip(train_confs, gt_confs):
                loss = criterion(torch.tensor(train_conf) * power.ruler_weight, torch.tensor(gt_conf).float())
                ruler_optimizer.zero_grad()
                loss.backward()
                ruler_optimizer.step()


def get_defaultdict():
    return defaultdict(list)


if __name__ == '__main__':
    main()
