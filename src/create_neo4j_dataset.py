import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle

from data.irt.split.split_dir import SplitDir
from data.neo4j.entities_tsv import Entity
from data.neo4j.facts_tsv import Fact
from data.neo4j.neo4j_dir import Neo4jDir
from data.neo4j.relations_tsv import Relation


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    create_neo4j_dataset(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) IRT Split Directory')

    parser.add_argument('neo4j_dir', metavar='neo4j-dir',
                        help='Path to (output) Neo4j Directory')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('neo4j-dir', args.neo4j_dir))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def create_neo4j_dataset(args):
    split_dir_path = args.split_dir
    neo4j_dir_path = args.neo4j_dir

    overwrite = args.overwrite

    #
    # Check that (input) IRT Split Directory exists
    #

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Create that (output) Neo4j Directory does not exist
    #

    neo4j_dir = Neo4jDir(Path(neo4j_dir_path))
    neo4j_dir.create(overwrite=overwrite)

    #
    # Create dataset
    #

    # Load ent/rel labels
    ent_to_lbl = split_dir.ent_labels_txt.load()
    rel_to_lbl = split_dir.rel_labels_txt.load()

    # Create Neo4j Entities TSV
    entities = [Entity(ent, lbl) for ent, lbl in ent_to_lbl.items()]
    neo4j_dir.entities_tsv.save(entities)

    # Create Neo4j Relations TSV
    relations = [Relation(rel, lbl) for rel, lbl in rel_to_lbl.items()]
    neo4j_dir.relations_tsv.save(relations)

    #
    # Create Neo4j Train Facts TSV
    #

    cw_train_triples = split_dir.cw_train_triples_txt.load()
    cw_valid_triples = split_dir.cw_valid_triples_txt.load()

    cw_triples = cw_train_triples + cw_valid_triples
    shuffle(cw_triples)

    train_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                   for head, rel, tail in cw_triples]

    neo4j_dir.train_facts_tsv.save(train_facts)

    #
    # Create Neo4j Valid Facts TSVs
    #

    ow_valid_triples = split_dir.ow_valid_triples_txt.load()
    shuffle(ow_valid_triples)

    valid_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                   for head, rel, tail in ow_valid_triples]

    valid_facts_count = len(valid_facts)
    v25 = valid_facts_count // 4
    v50 = valid_facts_count // 2
    v75 = valid_facts_count * 3 // 4
    
    neo4j_dir.valid_facts_25_1_tsv.save(valid_facts[:v25])
    neo4j_dir.valid_facts_25_2_tsv.save(valid_facts[v25:v50])
    neo4j_dir.valid_facts_25_3_tsv.save(valid_facts[v50:v75])
    neo4j_dir.valid_facts_25_4_tsv.save(valid_facts[v75:])

    #
    # Create Neo4j Test Facts TSVs
    #

    ow_test_triples = split_dir.ow_test_triples_txt.load()
    shuffle(ow_test_triples)

    test_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                   for head, rel, tail in ow_test_triples]

    test_facts_count = len(test_facts)
    t25 = test_facts_count // 4
    t50 = test_facts_count // 2
    t75 = test_facts_count * 3 // 4
    
    neo4j_dir.test_facts_25_1_tsv.save(test_facts[:t25])
    neo4j_dir.test_facts_25_2_tsv.save(test_facts[t25:t50])
    neo4j_dir.test_facts_25_3_tsv.save(test_facts[t50:t75])
    neo4j_dir.test_facts_25_4_tsv.save(test_facts[t75:])


if __name__ == '__main__':
    main()
