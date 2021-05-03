import logging
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

from data.power.samples.samples_dir import SamplesDir
from data.power.split.split_dir import SplitDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    eval_zero_rule(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('samples_dir', metavar='samples-dir',
                        help='Path to (input) POWER Samples Directory')

    parser.add_argument('class_count', metavar='class-count', type=int,
                        help='Number of classes distinguished by the classifier')

    parser.add_argument('sent_count', metavar='sent-count', type=int,
                        help='Number of sentences per entity')

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) POWER Split Directory')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('samples-dir', args.samples_dir))
    logging.info('    {:24} {}'.format('class-count', args.class_count))
    logging.info('    {:24} {}'.format('sent-count', args.sent_count))
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('--random-seed', args.random_seed))

    return args


def eval_zero_rule(args):
    samples_dir_path = args.samples_dir
    class_count = args.class_count
    sent_count = args.sent_count
    split_dir_path = args.split_dir

    #
    # Check that (input) POWER Samples Directory exists
    #

    logging.info('Check that (input) POWER Samples Directory exists ...')

    samples_dir = SamplesDir(Path(samples_dir_path))
    samples_dir.check()

    #
    # Check that (input) POWER Split Directory exists
    #

    logging.info('Check that (input) POWER Split Directory exists ...')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Load entity/relation labels
    #

    logging.info('Load entity/relation labels ...')

    ent_to_lbl = split_dir.entities_tsv.load()
    rel_to_lbl = split_dir.relations_tsv.load()

    #
    # Load datasets
    #

    logging.info('Load test dataset ...')

    test_set = samples_dir.test_samples_tsv.load(class_count, sent_count)

    #
    # Calc class frequencies
    #

    logging.info('Calc class frequencies ...')

    _, _, test_classes_stack, _ = zip(*test_set)
    test_freqs = np.array(test_classes_stack).mean(axis=0)

    #
    # Evaluate
    #

    logging.info(f'test_freqs = {test_freqs}')

    for strategy in ('uniform', 'stratified', 'most_frequent', 'constant'):
        logging.info(strategy)

        mean_metrics = []
        for i, gt in tqdm(enumerate(np.array(test_classes_stack).T)):

            if strategy == 'constant':
                classifier = DummyClassifier(strategy='constant', constant=1)
                classifier.fit([0, 1], [0, 1])
            else:
                classifier = DummyClassifier(strategy=strategy)
                classifier.fit(gt, gt)

            metrics_list = []
            for _ in range(10):
                pred = classifier.predict(gt)

                acc = accuracy_score(gt, pred)
                prec, recall, f1, _ = precision_recall_fscore_support(gt, pred, labels=[1], zero_division=1)

                metrics_list.append((acc, prec[0], recall[0], f1[0]))

            mean_metrics.append(np.mean(metrics_list, axis=0))

        logging.info(mean_metrics[0])
        logging.info(mean_metrics[-1])
        logging.info(np.mean(mean_metrics, axis=0))


if __name__ == '__main__':
    main()
