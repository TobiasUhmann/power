import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch import tensor, Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vocab

from dao.ower.ower_dir import OwerDir
from dao.ower.ower_samples_tsv import SamplesTsv
from ower.classifier import Classifier


def main():
    #
    # Parse args
    #

    parser = ArgumentParser()

    parser.add_argument('ower_dataset_dir', metavar='ower-dataset-dir',
                        help='Path to (input) OWER Dataset Directory')

    parser.add_argument('class_count', metavar='class-count', type=int,
                        help='Number of classes distinguished by the classifier')

    parser.add_argument('sent_count', metavar='sent-count', type=int,
                        help='Number of sentences per entity')

    device_choices = ['cpu', 'cuda']
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', metavar='STR', choices=device_choices, default=default_device,
                        help='Where to perform tensor operations, one of {} (default: {})'.format(
                            device_choices, default_device))

    default_emb_size = 32
    parser.add_argument('--emb-size', dest='emb_size', type=int, metavar='INT', default=default_emb_size,
                        help='Embedding size for sentence and class embeddings (default: {})'.format(default_emb_size))

    default_epoch_count = 10
    parser.add_argument('--epoch-count', dest='epoch_count', type=int, metavar='INT', default=default_epoch_count,
                        help='Number of training epochs (default: {})'.format(default_epoch_count))

    default_learning_rate = 0.01
    parser.add_argument('--lr', dest='lr', type=float, metavar='FLOAT', default=default_learning_rate,
                        help='Learning rate (default: {})'.format(default_learning_rate))

    args = parser.parse_args()

    #
    # Print applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ower-dataset-dir', args.ower_dataset_dir))
    logging.info('    {:24} {}'.format('class-count', args.class_count))
    logging.info('    {:24} {}'.format('sent-count', args.sent_count))
    logging.info('')
    logging.info('    {:24} {}'.format('--device', args.device))
    logging.info('    {:24} {}'.format('--emb-size', args.emb_size))
    logging.info('    {:24} {}'.format('--epoch-count', args.epoch_count))
    logging.info('    {:24} {}'.format('--lr', args.lr))
    logging.info('')

    #
    # Check that OWER Directory exists
    #

    ower_dir = OwerDir('OWER Directory', Path(args.ower_dataset_dir))
    ower_dir.check()

    #
    # Run actual program
    #

    train_classifier(ower_dir, args.class_count, args.sent_count, args.device, args.emb_size, args.epoch_count, args.lr)


def train_classifier(ower_dir: OwerDir, class_count: int, sent_count: int, device: str, emb_size: int, epoch_count: int,
                     lr: float) -> None:

    train_set, valid_set, _test_set, vocab = load_datasets(str(ower_dir._path), class_count, sent_count)

    train_loader = DataLoader(train_set, batch_size=64, collate_fn=generate_batch, shuffle=True)
    valid_loader = DataLoader(train_set, batch_size=64, collate_fn=generate_batch)

    classifier = Classifier(vocab_size=len(vocab), emb_size=emb_size, class_count=class_count).to(device)
    optimizer = Adam(classifier.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()

    for epoch in range(epoch_count):
        running_loss = 0.0

        for batch in train_loader:
            inputs_batch, labels_batch = batch
            inputs_batch = inputs_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs_batch = classifier(inputs_batch)
            loss = criterion(outputs_batch, labels_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            logging.info(running_loss)


@dataclass
class Sample:
    ent: int
    classes: List[int]
    sents: List[List[int]]


def load_datasets(ower_dir: Dict[str], class_count: int, sent_count: int) \
        -> Tuple[List[Sample], List[Sample], List[Sample], Vocab]:
    """
    Read train/valid/test Sample TSVs from OWER Directory and return rows as lists
    of samples. The sentences are tokenized by whitespace and mapped to IDs. The
    first sentence column of the train Samples TSV is used to build the vocabulary.

    Example row (tabs are represented by 4 spaces) and resulting sample:
    123    1    1    1    0    Barack Obama is male.    Barack Obama is married.
    123, [1, 1, 1, 0], [[1, 2, 3, 4], [1, 2, 3, 5]]
    """

    #
    # Define columns for subsequent read into TabularDatasets
    #

    def tokenize(text: str):
        return text.split()

    ent_field = ('ent', Field(sequential=False, use_vocab=False))

    class_fields = [(f'class_{i}', Field(sequential=False, use_vocab=False))
                    for i in range(class_count)]

    sent_fields = [(f'sent_{i}', Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True))
                   for i in range(sent_count)]

    fields = [ent_field] + class_fields + sent_fields

    #
    # Read Samples TSVs into TabularDatasets
    #

    raw_train_set = TabularDataset(ower_dir['train_samples_tsv'], 'tsv', fields, skip_header=True)
    raw_valid_set = TabularDataset(ower_dir['valid_samples_tsv'], 'tsv', fields, skip_header=True)
    raw_test_set = TabularDataset(ower_dir['test_samples_tsv'], 'tsv', fields, skip_header=True)

    #
    # Build vocab
    #

    first_sent_field = sent_fields[0][1]
    first_sent_field.build_vocab(raw_train_set)
    vocab = first_sent_field.vocab

    #
    # TabularDataset -> List[Sample] (i.e. parse ints, map tokens -> IDs)
    #

    def transform(raw_set: TabularDataset) -> List[Sample]:
        return [Sample(
            int(getattr(row, 'ent')),
            [int(getattr(row, f'class_{i}')) for i in range(class_count)],
            [[vocab[token] for token in getattr(row, f'sent_{i}')] for i in range(sent_count)]
        ) for row in raw_set]

    train_set = transform(raw_train_set)
    valid_set = transform(raw_valid_set)
    test_set = transform(raw_test_set)

    return train_set, valid_set, test_set, vocab


def generate_batch(batch: List[Tuple[int, List[int], List[List[int]]]]) -> Tuple[Tensor, Tensor]:
    _ent, classes_batch, sents_batch, = zip(*batch)

    cropped_sents_batch = [[sent[:32]
                            for sent in sents] for sents in sents_batch]

    padded_sents_batch = [[sent + [0] * (32 - len(sent))
                           for sent in sents] for sents in cropped_sents_batch]

    return tensor(padded_sents_batch), tensor(classes_batch)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
    main()
