import logging
import os
import random
from argparse import ArgumentParser

from neo4j import GraphDatabase


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    load_neo4j_graph(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('url', metavar='url',
                        help='URL of running Neo4j instance')

    parser.add_argument('username', metavar='username',
                        help='Username for running Neo4j instance')

    parser.add_argument('password', metavar='password',
                        help='Password for running Neo4j instance')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('url', args.url))
    logging.info('    {:24} {}'.format('username', args.username))
    logging.info('    {:24} {}'.format('password', args.password))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def load_neo4j_graph(args):
    url = args.url
    username = args.username
    password = args.password

    overwrite = args.overwrite

    driver = GraphDatabase.driver(url, auth=(username, password))

    with driver.session() as session:
        #
        # Delete graph if --overwrite
        #

        if overwrite:
            logging.info(f'Delete graph ...')
            session.write_transaction(delete_entities)
            session.write_transaction(drop_entities_constraint)
            logging.info(f'Deleted graph')

        #
        # Build graph
        #

        session.write_transaction(create_entities_constraint)

        logging.info(f'Load entities ...')
        entities_count = session.write_transaction(load_entities_tsv, 'entities.tsv')
        logging.info(f'Loaded {entities_count} entities')

        logging.info(f'Load train facts ...')
        train_facts_count = session.write_transaction(load_facts_tsv, 'train.tsv', 'train')
        logging.info(f'Loaded {train_facts_count} train facts')

        logging.info(f'Load known valid facts ...')
        valid_facts_count = session.write_transaction(load_facts_tsv, 'valid_known.tsv', 'valid')
        logging.info(f'Loaded {valid_facts_count} known valid facts')


def delete_entities(tx):
    tx.run('MATCH (n) DETACH DELETE n')


def drop_entities_constraint(tx):
    tx.run('DROP CONSTRAINT UniqueEntityId IF EXISTS')


def create_entities_constraint(tx):
    tx.run('CREATE CONSTRAINT UniqueEntityId ON (e:Entity) ASSERT e.id IS UNIQUE')


def load_entities_tsv(tx, filename: str):
    cypher = '''
        LOAD CSV WITH HEADERS FROM 'file:///' + $filename AS row FIELDTERMINATOR '\t'
        MERGE (ent:Entity {id: toInteger(row.id), label: row.lbl})
        RETURN COUNT(*)
    '''

    record = tx.run(cypher, filename=filename).single()

    return record[0]


def load_facts_tsv(tx, filename: str, split: str):
    cypher = '''
        LOAD CSV WITH HEADERS FROM 'file:///' + $filename AS row FIELDTERMINATOR '\t'
        MATCH (head:Entity {id: toInteger(row.head)})
        MATCH (tail:Entity {id: toInteger(row.tail)})
        CALL apoc.merge.relationship(
            head,
            'R_' + row.rel,
            {id: toInteger(row.rel), label: row.rel_lbl, split: $split},
            {},
            tail
        ) YIELD rel
        RETURN COUNT(*)
    '''

    record = tx.run(cypher, filename=filename, split=split).single()

    return record[0]


if __name__ == '__main__':
    main()
