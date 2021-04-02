import logging
import os
import random
from argparse import ArgumentParser
from pathlib import Path

from data.neo4j.entities_tsv import Entity
from data.neo4j.facts_tsv import Fact
from data.neo4j.relations_tsv import Relation
from data.neo4j.neo4j_dir import Neo4jDir
from data.irt.split.split_dir import SplitDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    create_neo4j_graph(args)

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


def create_neo4j_graph(args):
    split_dir_path = args.split_dir
    neo4j_dir_path = args.neo4j_dir

    #
    # Check (input) IRT Split Directory
    #

    logging.info('Check (input) IRT Split Directory ...')


    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Create (output) Neo4j Directory
    #

    logging.info('Create (output) Neo4j Directory ...')

    neo4j_dir = Neo4jDir(Path(neo4j_dir_path))
    neo4j_dir.create()

    #
    # Create dataset
    #

    logging.info('Create dataset ...')

    # Load ent/rel labels
    ent_to_lbl = split_dir.ent_labels_txt.load()
    rel_to_lbl = split_dir.rel_labels_txt.load()

    # Create Neo4j Entities TSV
    entities = [Entity(ent, lbl) for ent, lbl in ent_to_lbl.items()]
    neo4j_dir.entities_tsv.save(entities)

    # Create Neo4j Relations TSV
    relations = [Relation(rel, lbl) for rel, lbl in rel_to_lbl.items()]
    neo4j_dir.relations_tsv.save(relations)

    # Create Neo4j CW Train Facts TSV
    cw_train_triples = split_dir.cw_train_triples_txt.load()
    cw_train_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                      for head, rel, tail in cw_train_triples]
    neo4j_dir.cw_train_facts_tsv.save(cw_train_facts)

    # Create Neo4j CW Valid Facts TSV
    cw_valid_triples = split_dir.cw_valid_triples_txt.load()
    cw_valid_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                      for head, rel, tail in cw_valid_triples]
    neo4j_dir.cw_valid_facts_tsv.save(cw_valid_facts)

    # Create Neo4j OW Valid Facts TSV
    ow_valid_triples = split_dir.ow_valid_triples_txt.load()
    ow_valid_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                      for head, rel, tail in ow_valid_triples]
    neo4j_dir.ow_valid_facts_tsv.save(ow_valid_facts)

    # Create Neo4j OW Test Facts TSV
    ow_test_triples = split_dir.ow_test_triples_txt.load()
    ow_test_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                     for head, rel, tail in ow_test_triples]
    neo4j_dir.ow_test_facts_tsv.save(ow_test_facts)


if __name__ == '__main__':
    main()
