import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

    train_classifier(ower_dir)


@dataclass
class Config:
    ower_dataset_dir: str


def parse_args() -> Config:
    parser = ArgumentParser()

    parser.add_argument('ower_dataset_dir', metavar='ower-dataset-dir',
                        help='Path to (input) OWER Dataset Directory')

    args = parser.parse_args()

    config = Config(args.ower_dataset_dir)

    return config


def print_config(config: Config) -> None:
    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ower-dataset-dir', config.ower_dataset_dir))
    logging.info('')


def train_classifier(ower_dir: OwerDir) -> None:
    train_loader, vocab = get_train_loader(ower_dir.train_samples_tsv, 4, 3)

    classifier = Classifier(vocab_size=len(vocab), emb_size=32, class_count=4)
    optimizer = Adam(classifier.parameters(), lr=0.01)
    criterion = BCEWithLogitsLoss()

    for epoch in range(10):
        running_loss = 0.0

        for batch in train_loader:
            inputs_batch, labels_batch = batch

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
