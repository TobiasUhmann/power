import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from neo4j import GraphDatabase

from data.anyburl.rules.rules_dir import RulesDir


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    train_ruler(args)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('rules_dir', metavar='rules-dir',
                        help='Path to (input) AnyBURL Rules Directory')

    parser.add_argument('model_dir', metavar='model-dir',
                        help='Path to (output) POWER Model Directory')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    args = parser.parse_args()

    #
    # Log applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('rules-dir', args.rules_dir))
    logging.info('    {:24} {}'.format('model-dir', args.model_dir))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def train_ruler(args):
    rules_dir_path = args.rules_dir
    model_dir_path = args.model_dir

    overwrite = args.overwrite

    #
    # Check (input) AnyBURL Rules Directory
    #

    logging.info('Check (input) AnyBURL Rules Directory ...')

    rules_dir = RulesDir(Path(rules_dir_path))
    rules_dir.check()

    #
    # Create (output) POWER Model Directory
    #

    logging.info('Create (output) POWER Model Directory ...')

    #
    # Read rules
    #

    rules = rules_dir.cw_train_rules_tsv.load()

    good_rules = [rule for rule in rules if rule.confidence > 0.8]
    good_rules.sort(key=lambda rule: rule.confidence, reverse=True)

    short_rules = [rule for rule in good_rules if len(rule.body) == 1]
    pprint(short_rules)

    #
    #
    #

    def query_heads(tx, rel: int, tail: int):
        cypher = f'''
            MATCH (head)-[:R_{rel}]->(tail)
            WHERE tail.id = $tail
            RETURN head
        '''

        result = tx.run(cypher, tail=tail)

        return [record for record in result]

    def query_tails(tx, head: int, rel: int):
        cypher = f'''
            MATCH (head)-[:R_{rel}]->(tail)
            WHERE head.id = $head
            RETURN tail
        '''

        result = tx.run(cypher, head=head)

        return [record for record in result]

    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '1234567890'))
    unsupported_rules = 0

    pred = defaultdict(list)

    with driver.session() as session:
        for rule in short_rules[:5]:

            print()
            pprint(rule)

            #
            # Process rule body
            #

            body_fact = rule.body[0]

            if type(body_fact.head) == str and type(body_fact.tail) == int:
                ents = session.write_transaction(query_heads, rel=body_fact.rel, tail=body_fact.tail)
                pprint(ents)

            elif type(body_fact.head) == int and type(body_fact.tail) == str:
                ents = session.write_transaction(query_tails, head=body_fact.head, rel=body_fact.rel)
                pprint(ents)

            else:
                logging.warning(f'Unsupported rule body in rule {rule}')
                unsupported_rules += 1
                continue

            #
            # Process rule head
            #

            head_fact = rule.head

            if type(head_fact.head) == str and type(head_fact.tail) == int:
                for ent in ents:
                    pred[ent].append(((head_fact.rel, head_fact.tail), rule))

            elif type(head_fact.head) == int and type(head_fact.tail) == int:
                for ent in ents:
                    pred[head_fact.head].append(((head_fact.rel, ent), rule))

            else:
                logging.warning(f'Unsupported rule head in rule {rule}')
                unsupported_rules += 1
                continue

    print()
    pprint(pred)

    driver.close()


if __name__ == '__main__':
    main()
