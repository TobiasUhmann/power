from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, path
from os.path import isdir, isfile
from sqlite3 import connect
from typing import List, Tuple, Dict, Set

from dao.classes_tsv import read_classes_tsv
from dao.contexts_txt import read_contexts_txt
from dao.ower.samples_tsv import write_samples_tsv
from dao.ower.triples_db import create_triples_table, insert_triple, DbTriple, select_entities_with_class
from dao.ryn.triples_txt import load_triples


def main() -> None:
    #
    # Parse args
    #

    parser = ArgumentParser()

    parser.add_argument('ryn_dataset_dir', metavar='ryn-dataset-dir',
                        help='Path to (input) Ryn Dataset Directory')

    parser.add_argument('classes_tsv', metavar='classes-tsv',
                        help='Path to (input) Classes TSV')

    parser.add_argument('num_sentences', metavar='num-sentences', type=int,
                        help='Number of sentences per entity, entities with less sentences'
                             ' will be dropped')

    parser.add_argument('ower_dataset_dir', metavar='ower-dataset-dir',
                        help='Path to (output) OWER Dataset Directory')

    args = parser.parse_args()

    ryn_dataset_dir = args.ryn_dataset_dir
    classes_tsv = args.classes_tsv
    num_sentences = args.num_sentences
    ower_dataset_dir = args.ower_dataset_dir

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:20} {}'.format('ryn-dataset-dir', ryn_dataset_dir))
    print('    {:20} {}'.format('classes-tsv', classes_tsv))
    print('    {:20} {}'.format('num-sentences', num_sentences))
    print('    {:20} {}'.format('ower-dataset-dir', ower_dataset_dir))
    print()

    #
    # Assert that (input) Ryn Dataset Directory exists
    #

    if not isdir(ryn_dataset_dir):
        print('Ryn Dataset Directory not found')
        exit()

    ryn_dataset_files = {
        'triples_train_txt': path.join(ryn_dataset_dir, 'split', 'cw.train2id.txt'),
        'triples_valid_txt': path.join(ryn_dataset_dir, 'split', 'ow.valid2id.txt'),
        'triples_test_txt': path.join(ryn_dataset_dir, 'split', 'ow.test2id.txt'),

        'contexts_train_txt': path.join(ryn_dataset_dir, 'text', 'cw.train-sentences.txt'),
        'contexts_valid_txt': path.join(ryn_dataset_dir, 'text', 'ow.valid-sentences.txt'),
        'contexts_test_txt': path.join(ryn_dataset_dir, 'text', 'ow.test-sentences.txt'),
    }

    #
    # Assert that (input) Classes TSV exists
    #

    if not isfile(classes_tsv):
        print('Classes TSV not found')
        exit()

    #
    # Create (output) OWER Dataset Directory if it does not exist already
    #

    makedirs(ower_dataset_dir, exist_ok=True)

    ower_dataset_files = {
        'triples_train_db': path.join(ower_dataset_dir, 'triples-v1-train.db'),
        'triples_valid_db': path.join(ower_dataset_dir, 'triples-v1-valid.db'),
        'triples_test_db': path.join(ower_dataset_dir, 'triples-v1-test.db'),

        'samples_train_tsv': path.join(ower_dataset_dir, 'samples-v2-train.tsv'),
        'samples_valid_tsv': path.join(ower_dataset_dir, 'samples-v2-valid.tsv'),
        'samples_test_tsv': path.join(ower_dataset_dir, 'samples-v2-test.tsv')
    }

    #
    # Run actual program
    #

    create_ower_dataset(ryn_dataset_files, classes_tsv, num_sentences, ower_dataset_files)


def create_ower_dataset(
        ryn_dataset_files: Dict[str, str],
        classes_tsv: str,
        num_sentences: int,
        ower_dataset_files: Dict[str, str]
) -> None:
    #
    # Load triples from Triples TXTs
    #

    print()
    print('Load triples from Triples TXTs...')

    train_triples: List[Tuple[int, int, int]] = load_triples(ryn_dataset_files['triples_train_txt'])
    valid_triples: List[Tuple[int, int, int]] = load_triples(ryn_dataset_files['triples_valid_txt'])
    test_triples: List[Tuple[int, int, int]] = load_triples(ryn_dataset_files['triples_test_txt'])

    print('Done')

    #
    # Save triples to Triples DBs
    #

    print()
    print('Save triples to Triples DBs...')

    with connect(ower_dataset_files['triples_train_db']) as conn:
        create_triples_table(conn)
        for triple in train_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    with connect(ower_dataset_files['triples_valid_db']) as conn:
        create_triples_table(conn)
        for triple in valid_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    with connect(ower_dataset_files['triples_test_db']) as conn:
        create_triples_table(conn)
        for triple in test_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    print('Done')

    #
    # Load contexts from Contexts TXTs
    #

    print()
    print('Load contexts from Contexts TXTs...')

    train_contexts: Dict[int, Set[str]] = read_contexts_txt(ryn_dataset_files['contexts_train_txt'])
    valid_contexts: Dict[int, Set[str]] = read_contexts_txt(ryn_dataset_files['contexts_valid_txt'])
    test_contexts: Dict[int, Set[str]] = read_contexts_txt(ryn_dataset_files['contexts_test_txt'])

    print('Done')

    #
    # Get classes for each entity
    #

    print()
    print('Load contexts from Contexts TXTs...')

    classes: List[Tuple[int, int]] = read_classes_tsv(classes_tsv)

    train_class_to_entities = defaultdict(set)
    valid_class_to_entities = defaultdict(set)
    test_class_to_entities = defaultdict(set)

    with connect(ower_dataset_files['triples_train_db']) as conn:
        for class_ in classes:
            train_class_to_entities[class_] = select_entities_with_class(conn, class_)

    with connect(ower_dataset_files['triples_valid_db']) as conn:
        for class_ in classes:
            valid_class_to_entities[class_] = select_entities_with_class(conn, class_)

    with connect(ower_dataset_files['triples_test_db']) as conn:
        for class_ in classes:
            test_class_to_entities[class_] = select_entities_with_class(conn, class_)

    #
    # Save OWER TSVs
    #

    print()
    print('Save OWER TSVs...')

    train_tsv_rows = []
    valid_tsv_rows = []
    test_tsv_rows = []

    for ent in train_contexts:
        train_tsv_row = [ent]
        for class_ in classes:
            train_tsv_row.append(int(ent in train_class_to_entities[class_]))
        sentences = list(train_contexts[ent])[:num_sentences]
        if len(sentences) < num_sentences:
            continue
        train_tsv_row.append(sentences)
        train_tsv_rows.append(train_tsv_row)

    for ent in valid_contexts:
        valid_tsv_row = [ent]
        for class_ in classes:
            valid_tsv_row.append(int(ent in valid_class_to_entities[class_]))
        sentences = list(valid_contexts[ent])[:num_sentences]
        if len(sentences) < num_sentences:
            continue
        valid_tsv_row.append(sentences)
        valid_tsv_rows.append(valid_tsv_row)

    for ent in test_contexts:
        test_tsv_row = [ent]
        for class_ in classes:
            test_tsv_row.append(int(ent in test_class_to_entities[class_]))
        sentences = list(test_contexts[ent])[:num_sentences]
        if len(sentences) < num_sentences:
            continue
        test_tsv_row.append(sentences)
        test_tsv_rows.append(test_tsv_row)

    write_samples_tsv(ower_dataset_files['samples_train_tsv'], train_tsv_rows)
    write_samples_tsv(ower_dataset_files['samples_valid_tsv'], valid_tsv_rows)
    write_samples_tsv(ower_dataset_files['samples_test_tsv'], test_tsv_rows)

    print('Done')


if __name__ == '__main__':
    main()
