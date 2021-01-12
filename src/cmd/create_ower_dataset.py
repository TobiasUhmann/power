from argparse import ArgumentParser
from os.path import isdir
from typing import List, Tuple

from dao.triples_txt import load_triples


def main() -> None:
    #
    # Parse args
    #

    parser = ArgumentParser()

    parser.add_argument('ryn_dataset_dir', metavar='ryn-dataset-dir',
                        help='Path to (input) Ryn Dataset Directory')

    args = parser.parse_args()

    ryn_dataset_dir = args.ryn_dataset_dir

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:20} {}'.format('ryn-dataset-dir', ryn_dataset_dir))
    print()

    #
    # Assert that (input) Ryn Dataset Directory exists
    #

    if not isdir(ryn_dataset_dir):
        print('Ryn Dataset Directory not found')
        exit()

    #
    # Run actual program
    #

    create_ower_dataset(ryn_dataset_dir)


def create_ower_dataset(ryn_dataset_dir: str) -> None:
    #
    # Load triples from Triples TXTs
    #

    print()
    print('Load triples...')

    train_triples_file = f'{ryn_dataset_dir}/split/cw.train2id.txt'
    valid_triples_file = f'{ryn_dataset_dir}/split/ow.valid2id.txt'
    test_triples_file = f'{ryn_dataset_dir}/split/ow.test2id.txt'

    train_triples: List[Tuple[int, int, int]] = load_triples(train_triples_file)
    valid_triples: List[Tuple[int, int, int]] = load_triples(valid_triples_file)
    test_triples: List[Tuple[int, int, int]] = load_triples(test_triples_file)

    print('Done')


if __name__ == '__main__':
    main()
