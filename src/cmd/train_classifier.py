import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
    config: Config = parse_args()
    print_config(config)

    # Check that OWER Directory exists
    ower_dir = OwerDir('OWER Directory', Path(config.ower_dataset_dir))
    ower_dir.check()

    train_classifier(ower_dir,
                     config.class_count,
                     config.sent_count,
                     config.epoch_count,
                     config.device)


@dataclass
class Config:
    ower_dataset_dir: str
    class_count: int
    sent_count: int

    device: str
    epoch_count: int


def parse_args() -> Config:
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

    default_epoch_count = 10
    parser.add_argument('--epoch-count', dest='epoch_count', type=int, metavar='INT',
                        default=default_epoch_count,
                        help='Number of training epochs (default: {})'.format(default_epoch_count))

    args = parser.parse_args()

    config = Config(args.ower_dataset_dir,
                    args.class_count,
                    args.sent_count,
                    args.device,
                    args.epoch_count)

    return config


def print_config(config: Config) -> None:
    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ower-dataset-dir', config.ower_dataset_dir))
    logging.info('    {:24} {}'.format('class-count', config.class_count))
    logging.info('    {:24} {}'.format('sent-count', config.sent_count))
    logging.info('')
    logging.info('    {:24} {}'.format('--device', config.device))
    logging.info('    {:24} {}'.format('--epoch-count', config.epoch_count))
    logging.info('')


def train_classifier(ower_dir: OwerDir,
                     class_count: int,
                     sent_count: int,
                     epoch_count: int,
                     device: str
                     ) -> None:
    train_loader, vocab = get_train_loader(ower_dir.train_samples_tsv, class_count, sent_count)

    classifier = Classifier(vocab_size=len(vocab), emb_size=32, class_count=4).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01)
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


def get_train_loader(train_samples_tsv: SamplesTsv, class_count: int, sent_count: int) -> Tuple[DataLoader, Vocab]:
    def tokenize(text: str):
        return text.split()

    ent_field = ('ent', Field(sequential=False, use_vocab=False))
    class_fields = [(f'class_{i}', Field(sequential=False, use_vocab=False))
                    for i in range(class_count)]
    sent_fields = [(f'sent_{i}', Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True))
                   for i in range(sent_count)]

    fields = [ent_field] + class_fields + sent_fields

    raw_train_set = TabularDataset(str(train_samples_tsv._path), 'tsv', fields, skip_header=True)

    first_sent_field = sent_fields[0][1]
    first_sent_field.build_vocab(raw_train_set)
    vocab = first_sent_field.vocab

    train_set = [(
        int(getattr(sample, 'ent')),
        [int(getattr(sample, f'class_{i}')) for i in range(class_count)],
        [[vocab[token] for token in getattr(sample, f'sent_{i}')] for i in range(sent_count)]
    ) for sample in raw_train_set]

    return DataLoader(train_set, batch_size=64, collate_fn=generate_batch, shuffle=True), vocab


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
