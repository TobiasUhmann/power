from argparse import ArgumentParser
from os.path import isdir


def main():
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

    create_ower_dataset()


def create_ower_dataset():
    pass


if __name__ == '__main__':
    main()
