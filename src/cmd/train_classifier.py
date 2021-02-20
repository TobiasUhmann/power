import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dao.ower.ower_dir import OwerDir


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
    train_loader = DataLoader()
    classifier = Module()
    optimizer = Adam(classifier.parameters(), lr=0.01)
    criterion = BCEWithLogitsLoss()

    for epoch in range(10):
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            inputs_batch, labels_batch = batch

            outputs_batch = classifier(inputs_batch)
            loss = criterion(outputs_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
    main()
