from collections import defaultdict
from sqlite3 import connect
from typing import Dict, Set, List, Tuple

from tqdm import tqdm

from dao.contexts_txt import load_contexts
from dao.output_txt import save_outputs
from dao.triples_db import create_triples_table, insert_triple, DbTriple, select_triples_by_head_rel_and_tail
from dao.triples_txt import load_triples


def main():
    dataset_dir = '../data/irt.fb.30.26041992.clean/'

    #
    # Load triples from TXTs
    #

    print()
    print('Load triples...')

    train_triples_file = f'{dataset_dir}/split/cw.train2id.txt'
    valid_triples_file = f'{dataset_dir}/split/ow.valid2id.txt'
    test_triples_file = f'{dataset_dir}/split/ow.test2id.txt'

    train_triples: List[Tuple[int, int, int]] = load_triples(train_triples_file)
    valid_triples: List[Tuple[int, int, int]] = load_triples(valid_triples_file)
    test_triples: List[Tuple[int, int, int]] = load_triples(test_triples_file)

    print('Done')

    #
    # Save triples in DBs
    #

    with connect('../data/train.db') as conn:
        create_triples_table(conn)
        for triple in train_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    with connect('../data/valid.db') as conn:
        create_triples_table(conn)
        for triple in valid_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    with connect('../data/test.db') as conn:
        create_triples_table(conn)
        for triple in test_triples:
            insert_triple(conn, DbTriple(triple[0], triple[1], triple[2]))

    #
    # Load contexts from TXTs
    #

    print()
    print('Load contexts...')

    train_contexts_file = f'{dataset_dir}/text/cw.train-sentences.txt'
    valid_contexts_file = f'{dataset_dir}/text/ow.valid-sentences.txt'
    test_contexts_file = f'{dataset_dir}/text/ow.test-sentences.txt'

    train_contexts: Dict[int, Set[str]] = load_contexts(train_contexts_file)
    valid_contexts: Dict[int, Set[str]] = load_contexts(valid_contexts_file)
    test_contexts: Dict[int, Set[str]] = load_contexts(test_contexts_file)

    print('Done')

    #
    # Get label (is male?) for each entity
    #

    train_labels_male = defaultdict(bool)
    train_labels_married = defaultdict(bool)
    train_labels_american = defaultdict(bool)
    train_labels_actor = defaultdict(bool)
    with connect('../data/train.db') as conn:
        for ent in tqdm(train_contexts):
            is_male_triples = select_triples_by_head_rel_and_tail(conn, ent, 43, 141)
            is_married_triples = select_triples_by_head_rel_and_tail(conn, ent, 85, 434)
            is_american_triples = select_triples_by_head_rel_and_tail(conn, ent, 48, 32)
            is_actor = select_triples_by_head_rel_and_tail(conn, ent, 17, 862)

            if len(is_male_triples) == 1:
                train_labels_male[ent] = True

            if len(is_married_triples) == 1:
                train_labels_married[ent] = True

            if len(is_american_triples) == 1:
                train_labels_american[ent] = True

            if len(is_actor) == 1:
                train_labels_actor[ent] = True

    valid_labels_male = defaultdict(bool)
    valid_labels_married = defaultdict(bool)
    valid_labels_american = defaultdict(bool)
    valid_labels_actor = defaultdict(bool)
    with connect('../data/valid.db') as conn:
        for ent in tqdm(valid_contexts):
            is_male_triples = select_triples_by_head_rel_and_tail(conn, ent, 43, 141)
            is_married_triples = select_triples_by_head_rel_and_tail(conn, ent, 85, 434)
            is_american_triples = select_triples_by_head_rel_and_tail(conn, ent, 48, 32)
            is_actor = select_triples_by_head_rel_and_tail(conn, ent, 17, 862)

            if len(is_male_triples) == 1:
                valid_labels_male[ent] = True

            if len(is_married_triples) == 1:
                valid_labels_married[ent] = True

            if len(is_american_triples) == 1:
                valid_labels_american[ent] = True

            if len(is_actor) == 1:
                valid_labels_actor[ent] = True

    test_labels_male = defaultdict(bool)
    test_labels_married = defaultdict(bool)
    test_labels_american = defaultdict(bool)
    test_labels_actor = defaultdict(bool)
    with connect('../data/test.db') as conn:
        for ent in tqdm(test_contexts):
            is_male_triples = select_triples_by_head_rel_and_tail(conn, ent, 43, 141)
            is_married_triples = select_triples_by_head_rel_and_tail(conn, ent, 85, 434)
            is_american_triples = select_triples_by_head_rel_and_tail(conn, ent, 48, 32)
            is_actor = select_triples_by_head_rel_and_tail(conn, ent, 17, 862)

            if len(is_male_triples) == 1:
                test_labels_male[ent] = True

            if len(is_married_triples) == 1:
                test_labels_married[ent] = True

            if len(is_american_triples) == 1:
                test_labels_american[ent] = True

            if len(is_actor) == 1:
                test_labels_actor[ent] = True

    #
    # Write output file
    #

    train_outputs_txt = '../data/train_outputs.tsv'
    train_outputs: List[Tuple[int, int, int, int, int, str]] = []
    for ent in train_contexts:
        train_outputs.append((ent, 
                              int(train_labels_male[ent]), 
                              int(train_labels_married[ent]), 
                              int(train_labels_american[ent]), 
                              int(train_labels_actor[ent]), 
                              list(train_contexts[ent])[0].strip()))
    save_outputs(train_outputs_txt, train_outputs)

    valid_outputs_txt = '../data/valid_outputs.tsv'
    valid_outputs: List[Tuple[int, int, int, int, int, str]] = []
    for ent in valid_contexts:
        valid_outputs.append((ent,
                              int(valid_labels_male[ent]),
                              int(valid_labels_married[ent]),
                              int(valid_labels_american[ent]),
                              int(valid_labels_actor[ent]),
                              list(valid_contexts[ent])[0].strip()))
    save_outputs(valid_outputs_txt, valid_outputs)

    test_outputs_txt = '../data/test_outputs.tsv'
    test_outputs: List[Tuple[int, int, int, int, int, str]] = []
    for ent in test_contexts:
        test_outputs.append((ent,
                              int(test_labels_male[ent]),
                              int(test_labels_married[ent]),
                              int(test_labels_american[ent]),
                              int(test_labels_actor[ent]),
                              list(test_contexts[ent])[0].strip()))
    save_outputs(test_outputs_txt, test_outputs)


if __name__ == '__main__':
    main()
