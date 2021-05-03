import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List

from neo4j import GraphDatabase

from data.anyburl.rules_tsv import RulesTsv
from data.power.ruler_pkl import RulerPkl
from data.power.split.split_dir import SplitDir
from models.ent import Ent
from models.fact import Fact
from models.rel import Rel
from models.rule import Rule
from models.var import Var
from power.ruler import Ruler


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.INFO)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    prepare_ruler(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('rules_tsv', metavar='rules-tsv',
                        help='Path to (input) AnyBURL Rules TSV')

    parser.add_argument('url', metavar='url',
                        help='URL of running Neo4j instance')

    parser.add_argument('username', metavar='username',
                        help='Username for running Neo4j instance')

    parser.add_argument('password', metavar='password',
                        help='Password for running Neo4j instance')

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) POWER Split Directory')

    parser.add_argument('ruler_pkl', metavar='ruler-pkl',
                        help='Path to (output) POWER Ruler PKL')

    default_min_conf = 0.5
    parser.add_argument('--min-conf', dest='min_conf', type=float, metavar='FLOAT', default=default_min_conf,
                        help='Minimum confidence rules need to be considered (default:{})'.format(default_min_conf))

    default_min_supp = 1
    parser.add_argument('--min-supp', dest='min_supp', type=int, metavar='INT', default=default_min_supp,
                        help='Minimum support rules need to be considered (default:{})'.format(default_min_supp))

    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite output files if they already exist')

    parser.add_argument('--random-seed', dest='random_seed', metavar='STR',
                        help='Use together with PYTHONHASHSEED for reproducibility')

    return parser.parse_args()


def log_config(args) -> None:
    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('rules-tsv', args.rules_tsv))
    logging.info('    {:24} {}'.format('url', args.url))
    logging.info('    {:24} {}'.format('username', args.username))
    logging.info('    {:24} {}'.format('password', args.password))
    logging.info('    {:24} {}'.format('split-dir', args.split_dir))
    logging.info('    {:24} {}'.format('ruler-pkl', args.ruler_pkl))
    logging.info('    {:24} {}'.format('--min-conf', args.min_conf))
    logging.info('    {:24} {}'.format('--min-supp', args.min_supp))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))
    logging.info('    {:24} {}'.format('--random-seed', args.random_seed))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))


def prepare_ruler(args):
    rules_tsv_path = args.rules_tsv
    url = args.url
    username = args.username
    password = args.password
    split_dir_path = args.split_dir
    ruler_pkl_path = args.ruler_pkl

    min_conf = args.min_conf
    min_supp = args.min_supp
    overwrite = args.overwrite

    #
    # Check that (input) POWER Rules TSV exists
    #

    logging.info('Check that (input) POWER Rules TSV exists ...')

    rules_tsv = RulesTsv(Path(rules_tsv_path))
    rules_tsv.check()

    #
    # Check that (input) POWER Split Directory exists
    #

    logging.info('Check that (input) POWERT Split Directory exists ...')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Check that (output) POWER Ruler PKL does not exist
    #

    logging.info('Check that (output) POWER Ruler PKL does not exist ...')

    ruler_pkl = RulerPkl(Path(ruler_pkl_path))
    ruler_pkl.check(should_exist=overwrite)

    #
    # Read rules
    #

    logging.info('Read rules ...')

    ent_to_lbl = split_dir.entities_tsv.load()
    rel_to_lbl = split_dir.relations_tsv.load()

    anyburl_rules = rules_tsv.load()
    rules = [Rule.from_anyburl(rule, ent_to_lbl, rel_to_lbl) for rule in anyburl_rules]

    good_rules = [rule for rule in rules if rule.conf > min_conf and rule.fires > min_supp]
    good_rules.sort(key=lambda rule: rule.conf, reverse=True)

    short_rules = [rule for rule in good_rules if len(rule.body) == 1]
    log_rules('Rules', short_rules)

    #
    # Load train facts
    #

    logging.info('Load train facts ...')

    train_triples = split_dir.train_facts_tsv.load()
    train_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                   for head, _, rel, _, tail, _ in train_triples}

    #
    # Process rules
    #

    logging.info('Process rules ...')

    driver = GraphDatabase.driver(url, auth=(username, password))
    unsupported_rules = 0

    pred = defaultdict(get_defaultdict)

    with driver.session() as session:
        for rule in short_rules:

            logging.info(f'Process rule {rule}')

            #
            # Process rule body
            #

            body_fact = rule.body[0]

            if type(body_fact.head) == Var and type(body_fact.tail) == Ent:
                records = session.write_transaction(query_facts_by_rel_tail, rel=body_fact.rel, tail=body_fact.tail)
                ents = [Ent(head['id'], ent_to_lbl[head['id']]) for head, _, _ in records]

            elif type(body_fact.head) == Ent and type(body_fact.tail) == Var:
                records = session.write_transaction(query_facts_by_head_rel, head=body_fact.head, rel=body_fact.rel)
                ents = [Ent(tail['id'], ent_to_lbl[tail['id']]) for _, _, tail in records]

            else:
                logging.warning(f'Unsupported rule body in rule {rule}. Skipping.')
                unsupported_rules += 1
                continue

            #
            # Process rule head
            #

            head_fact = rule.head

            if type(head_fact.head) == Var and type(head_fact.tail) == Ent:
                pred_facts = [Fact(ent, head_fact.rel, head_fact.tail) for ent in ents]

            elif type(head_fact.head) == Ent and type(head_fact.tail) == Var:
                pred_facts = [Fact(head_fact.head, head_fact.rel, ent) for ent in ents]

            else:
                logging.warning(f'Unsupported rule head in rule {rule}. Skipping.')
                unsupported_rules += 1
                continue

            #
            # Filter out train facts and save predicted valid facts
            #

            for fact in pred_facts:
                # if fact not in train_facts:
                pred[fact.head][(fact.rel, fact.tail)].append(rule)

    driver.close()

    #
    # Persist ruler
    #

    logging.info('Persist ruler ...')

    ruler = Ruler()
    ruler.pred = pred

    ruler_pkl.save(ruler)


def get_defaultdict():
    return defaultdict(list)


def query_facts_by_rel_tail(tx, rel: Rel, tail: Ent):
    cypher = f'''
        MATCH (head)-[rel:R_{rel.id}]->(tail)
        WHERE tail.id = $tail_id
        RETURN head, rel, tail
    '''

    records = tx.run(cypher, tail_id=tail.id)

    return list(records)


def query_facts_by_head_rel(tx, head: Ent, rel: Rel):
    cypher = f'''
        MATCH (head)-[rel:R_{rel.id}]->(tail)
        WHERE head.id = $head_id
        RETURN head, rel, tail
    '''

    records = tx.run(cypher, head_id=head.id)

    return list(records)


def log_rules(msg: str, rules: List[Rule], display_max=10):
    if logging.getLogger().level == logging.DEBUG:

        logging.debug(f'{msg} ({display_max}/{len(rules)}):')
        for rule in rules[:display_max]:
            logging.debug(str(rule))


if __name__ == '__main__':
    main()
