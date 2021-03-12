"""
Create an `OWER Dataset` from a `Ryn Dataset`
"""

import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Set

from dao.ower.ower_dir import OwerDir
from dao.ryn.ryn_dir import RynDir


def main() -> None:
    """ Parse args, log config, check files and run program"""

    ## Parse args

    parser = ArgumentParser()

    parser.add_argument('ryn_dir_path', metavar='ryn-dir',
                        help='Path to (input) Ryn Directory')

    parser.add_argument('ower_dir_path', metavar='ower-dir',
                        help='Path to (output) OWER Directory')

    default_class_count = 100
    parser.add_argument('--class-count', dest='class_count', type=int, metavar='INT', default=default_class_count,
                        help='Number of classes (default: {})'.format(default_class_count))

    default_sent_count = 5
    parser.add_argument('--sent-count', dest='sent_count', type=int, metavar='INT', default=default_sent_count,
                        help='Number of sentences per entity. Entities for which not enough sentences'
                             ' are availabe are dropped. (default: {})'.format(default_sent_count))

    args = parser.parse_args()

    ryn_dir_path = args.ryn_dir_path
    ower_dir_path = args.ower_dir_path

    class_count = args.class_count
    sent_count = args.sent_count

    ## Log applied config

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ryn-dir', ryn_dir_path))
    logging.info('    {:24} {}'.format('ower-dir', ower_dir_path))
    logging.info('')
    logging.info('    {:24} {}'.format('--class-count', class_count))
    logging.info('    {:24} {}'.format('--sent-count', sent_count))
    logging.info('')

    ## Assert that (input) Ryn Directory exists

    ryn_dir = RynDir('Ryn Directory', Path(ryn_dir_path))
    ryn_dir.check()

    ## Create (output) OWER Directory if it does not exist yet

    ower_dir = OwerDir('OWER Directory', Path(ower_dir_path))
    ower_dir.create()

    ## Run actual program

    create_ower_dataset(ryn_dir, ower_dir, class_count, sent_count)


def create_ower_dataset(
        ryn_dir: RynDir,
        ower_dir: OwerDir,
        class_count: int,
        sent_count: int
) -> None:
    #
    # Load triples from Triples TXTs
    #

    print()
    print('Load triples from Triples TXTs...')

    split_dir = ryn_dir.split_dir
    cw_train_triples: List[Tuple[int, int, int]] = split_dir.cw_train_triples_txt.load()
    cw_valid_triples: List[Tuple[int, int, int]] = split_dir.cw_valid_triples_txt.load()
    ow_valid_triples: List[Tuple[int, int, int]] = split_dir.ow_valid_triples_txt.load()
    ow_test_triples: List[Tuple[int, int, int]] = split_dir.ow_test_triples_txt.load()

    train_triples = cw_train_triples + cw_valid_triples
    valid_triples = ow_valid_triples
    test_triples = ow_test_triples

    print('Done')

    #
    # Save triples to Triples DBs
    #

    print()
    print('Save triples to Triples DBs...')
    
    ower_dir.train_triples_db.create_triples_table()
    train_db_triples = [DbTriple(triple[0], triple[1], triple[2]) for triple in train_triples]
    ower_dir.train_triples_db.insert_triples(train_db_triples)
    
    ower_dir.valid_triples_db.create_triples_table()
    valid_db_triples = [DbTriple(triple[0], triple[1], triple[2]) for triple in valid_triples]
    ower_dir.valid_triples_db.insert_triples(valid_db_triples)
    
    ower_dir.test_triples_db.create_triples_table()
    test_db_triples = [DbTriple(triple[0], triple[1], triple[2]) for triple in test_triples]
    ower_dir.test_triples_db.insert_triples(test_db_triples)

    print('Done')

    #
    # Load contexts from Contexts TXTs
    #

    print()
    print('Load contexts from Contexts TXTs...')

    text_dir = ryn_dir.text_dir
    train_contexts: Dict[int, Set[str]] = text_dir.cw_train_sents_txt.load()
    valid_contexts: Dict[int, Set[str]] = text_dir.ow_valid_sents_txt.load()
    test_contexts: Dict[int, Set[str]] = text_dir.ow_test_sents_txt.load()

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
    
    for class_ in classes:
        train_class_to_entities[class_] = ower_dir.train_triples_db.select_entities_with_class(class_)
    
    for class_ in classes:
        valid_class_to_entities[class_] = ower_dir.valid_triples_db.select_entities_with_class(class_)
    
    for class_ in classes:
        test_class_to_entities[class_] = ower_dir.test_triples_db.select_entities_with_class(class_)

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
        
    ower_dir.train_samples_tsv.write_samples_tsv(train_tsv_rows)
    ower_dir.valid_samples_tsv.write_samples_tsv(valid_tsv_rows)
    ower_dir.test_samples_tsv.write_samples_tsv(test_tsv_rows)

    print('Done')


if __name__ == '__main__':
    main()
