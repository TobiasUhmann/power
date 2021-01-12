from argparse import ArgumentParser
from os import makedirs, path
from os.path import isdir
from sqlite3 import connect
from typing import List, Tuple, Dict, Set

from dao.contexts_txt import load_contexts
from dao.triples_db import create_triples_table, insert_triple, DbTriple
from dao.triples_txt import load_triples


def main() -> None:
    #
    # Parse args
    #

    parser = ArgumentParser()

    parser.add_argument('ryn_dataset_dir', metavar='ryn-dataset-dir',
                        help='Path to (input) Ryn Dataset Directory')

    default_work_dir = 'work/'
    parser.add_argument('--work-dir', metavar='STR', default=default_work_dir,
                        help='Path to (output) Working Directory (default: {})'.format(default_work_dir))

    args = parser.parse_args()

    ryn_dataset_dir = args.ryn_dataset_dir
    work_dir = args.work_dir

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:20} {}'.format('ryn-dataset-dir', ryn_dataset_dir))
    print()
    print('    {:20} {}'.format('--work-dir', work_dir))
    print()

    #
    # Assert that (input) Ryn Dataset Directory exists
    #

    if not isdir(ryn_dataset_dir):
        print('Ryn Dataset Directory not found')
        exit()

    train_triples_txt = path.join(ryn_dataset_dir, 'split', 'ow.train2id.txt')
    valid_triples_txt = path.join(ryn_dataset_dir, 'split', 'ow.valid2id.txt')
    test_triples_txt = path.join(ryn_dataset_dir, 'split', 'ow.test2id.txt')

    train_contexts_txt = path.join(ryn_dataset_dir, 'text', 'cw.train-sentences.txt')
    valid_contexts_txt = path.join(ryn_dataset_dir, 'text', 'cw.valid-sentences.txt')
    test_contexts_txt = path.join(ryn_dataset_dir, 'text', 'cw.test-sentences.txt')

    #
    # Create (output) Working Directory if it does not exist already
    #

    makedirs(work_dir, exist_ok=True)

    train_triples_db = path.join(work_dir, 'train_triples.db')
    valid_triples_db = path.join(work_dir, 'valid_triples.db')
    test_triples_db = path.join(work_dir, 'test_triples.db')

    #
    # Run actual program
    #

    create_ower_dataset(train_triples_txt, valid_triples_txt, test_triples_txt, train_contexts_txt, valid_contexts_txt,
                        test_contexts_txt, train_triples_db, valid_triples_db, test_triples_db)


def create_ower_dataset(
        train_triples_txt: str,
        valid_triples_txt: str,
        test_triples_txt: str,
        train_contexts_txt: str,
        valid_contexts_txt: str,
        test_contexts_txt: str,
        train_triples_db: str,
        valid_triples_db: str,
        test_triples_db: str
) -> None:
    #
    # Load triples from Triples TXTs
    #

    print()
    print('Load triples from Triples TXTs...')

    train_triples: List[Tuple[int, int, int]] = load_triples(train_triples_txt)
    valid_triples: List[Tuple[int, int, int]] = load_triples(valid_triples_txt)
    test_triples: List[Tuple[int, int, int]] = load_triples(test_triples_txt)

    print('Done')

    #
    # Save triples to Triples DBs
    #

    print()
    print('Save triples to Triples DBs...')

    with connect(train_triples_db) as conn:
        create_triples_table(conn)
        for triple in train_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    with connect(valid_triples_db) as conn:
        create_triples_table(conn)
        for triple in valid_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    with connect(test_triples_db) as conn:
        create_triples_table(conn)
        for triple in test_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    print('Done')

    #
    # Load contexts from Contexts TXTs
    #

    print()
    print('Load contexts from Contexts TXTs...')

    train_contexts: Dict[int, Set[str]] = load_contexts(train_contexts_txt)
    valid_contexts: Dict[int, Set[str]] = load_contexts(valid_contexts_txt)
    test_contexts: Dict[int, Set[str]] = load_contexts(test_contexts_txt)

    print('Done')


if __name__ == '__main__':
    main()
