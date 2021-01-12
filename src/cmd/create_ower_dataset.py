from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, path
from os.path import isdir, isfile
from sqlite3 import connect
from typing import List, Tuple, Dict, Set

from dao.classes_tsv import load_classes
from dao.contexts_txt import load_contexts
from dao.triples_db import create_triples_table, insert_triple, DbTriple, select_entities_with_class
from dao.triples_txt import load_triples


def main() -> None:
    #
    # Parse args
    #

    parser = ArgumentParser()

    parser.add_argument('ryn_dataset_dir', metavar='ryn-dataset-dir',
                        help='Path to (input) Ryn Dataset Directory')

    parser.add_argument('classes_tsv', metavar='classes-tsv',
                        help='Path to (input) Classes TSV')

    default_work_dir = 'work/'
    parser.add_argument('--work-dir', metavar='STR', default=default_work_dir,
                        help='Path to (output) Working Directory (default: {})'.format(default_work_dir))

    args = parser.parse_args()

    ryn_dataset_dir = args.ryn_dataset_dir
    classes_tsv = args.classes_tsv
    work_dir = args.work_dir

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:20} {}'.format('ryn-dataset-dir', ryn_dataset_dir))
    print('    {:20} {}'.format('classes-tsv', classes_tsv))
    print()
    print('    {:20} {}'.format('--work-dir', work_dir))
    print()

    #
    # Assert that (input) Ryn Dataset Directory exists
    #

    if not isdir(ryn_dataset_dir):
        print('Ryn Dataset Directory not found')
        exit()

    train_triples_txt = path.join(ryn_dataset_dir, 'split', 'cw.train2id.txt')
    valid_triples_txt = path.join(ryn_dataset_dir, 'split', 'ow.valid2id.txt')
    test_triples_txt = path.join(ryn_dataset_dir, 'split', 'ow.test2id.txt')

    train_contexts_txt = path.join(ryn_dataset_dir, 'text', 'cw.train-sentences.txt')
    valid_contexts_txt = path.join(ryn_dataset_dir, 'text', 'ow.valid-sentences.txt')
    test_contexts_txt = path.join(ryn_dataset_dir, 'text', 'ow.test-sentences.txt')

    #
    # Assert that (input) Classes TSV exists
    #

    if not isfile(classes_tsv):
        print('Classes TSV not found')
        exit()

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
                        test_contexts_txt, classes_tsv, train_triples_db, valid_triples_db, test_triples_db)


def create_ower_dataset(
        train_triples_txt: str,
        valid_triples_txt: str,
        test_triples_txt: str,
        train_contexts_txt: str,
        valid_contexts_txt: str,
        test_contexts_txt: str,
        classes_tsv: str,
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

    #
    # Get classes for each entity
    #

    print()
    print('Load contexts from Contexts TXTs...')

    classes: List[Tuple[int, int]] = load_classes(classes_tsv)

    train_class_to_entities = defaultdict(set)
    valid_class_to_entities = defaultdict(set)
    test_class_to_entities = defaultdict(set)

    with connect(train_triples_db) as conn:
        for class_ in classes:
            train_class_to_entities[class_] = select_entities_with_class(conn, class_)

    with connect(valid_triples_db) as conn:
        for class_ in classes:
            valid_class_to_entities[class_] = select_entities_with_class(conn, class_)

    with connect(test_triples_db) as conn:
        for class_ in classes:
            test_class_to_entities[class_] = select_entities_with_class(conn, class_)


if __name__ == '__main__':
    main()
