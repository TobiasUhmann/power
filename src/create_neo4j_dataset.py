import logging
from argparse import ArgumentParser
from pathlib import Path

from data.neo4j.entities_tsv import Entity
from data.neo4j.facts_tsv import Fact
from data.neo4j.relations_tsv import Relation
from data.neo4j.neo4j_dir import Neo4jDir
from data.ryn.split.split_dir import SplitDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    create_neo4j_dataset(args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) Ryn Split Directory')

    parser.add_argument('neo4j_dir', metavar='neo4j-dir',
                        help='Path to (output) Neo4j Directory')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('neo4j-dir', args.neo4j_dir))

    return args


def create_neo4j_dataset(args):
    split_dir_path = args.split_dir
    neo4j_dir_path = args.neo4j_dir

    #
    # Check that (input) OWER Directory exists
    #

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Create (output) Neo4j Directory if it does not exist already
    #

    neo4j_dir = Neo4jDir(Path(neo4j_dir_path))
    neo4j_dir.create()

    #
    # Load KG from Ryn Split Directory, transform, and save to Neo4j Directory
    #

    ent_to_lbl = split_dir.ent_labels_txt.load()
    entities = [Entity(ent, lbl) for ent, lbl in ent_to_lbl.items()]
    neo4j_dir.entities_tsv.save(entities)

    rel_to_lbl = split_dir.rel_labels_txt.load()
    relations = [Relation(rel, lbl) for rel, lbl in rel_to_lbl.items()]
    neo4j_dir.relations_tsv.save(relations)

    cw_train_triples = split_dir.cw_train_triples_txt.load()
    cw_train_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                      for head, rel, tail in cw_train_triples]
    neo4j_dir.cw_train_facts_tsv.save(cw_train_facts)

    cw_valid_triples = split_dir.cw_valid_triples_txt.load()
    cw_valid_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                      for head, rel, tail in cw_valid_triples]
    neo4j_dir.cw_valid_facts_tsv.save(cw_valid_facts)

    ow_valid_triples = split_dir.ow_valid_triples_txt.load()
    ow_valid_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                      for head, rel, tail in ow_valid_triples]
    neo4j_dir.ow_valid_facts_tsv.save(ow_valid_facts)

    ow_test_triples = split_dir.ow_test_triples_txt.load()
    ow_test_facts = [Fact(head, ent_to_lbl[head], rel, rel_to_lbl[rel], tail, ent_to_lbl[tail])
                     for head, rel, tail in ow_test_triples]
    neo4j_dir.ow_test_facts_tsv.save(ow_test_facts)


if __name__ == '__main__':
    main()
