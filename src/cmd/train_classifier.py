import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from dao.ower.ower_dir import OwerDir


def main():
    config: Config = parse_args()
    print_config(config)
    check_files(config)


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


def check_files(config: Config) -> None:
    ower_dir = OwerDir('OWER Directory', Path(config.ower_dataset_dir))
    ower_dir.check()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
    main()
