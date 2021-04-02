import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Set, List

from neo4j import GraphDatabase

from data.anyburl.rules.rules_dir import RulesDir
from data.power.model.model_dir import ModelDir
from data.irt.split.split_dir import SplitDir
from models.ent import Ent
from models.fact import Fact
from models.rel import Rel
from models.rule import Rule
from models.split import Split
from models.var import Var
from power.ruler import Ruler


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)-7s | %(message)s', level=logging.DEBUG)

    args = parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    train_ruler(args)

    logging.info('Finished successfully')


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('rules_dir', metavar='rules-dir',
                        help='Path to (input) AnyBURL Rules Directory')

    parser.add_argument('split_dir', metavar='split-dir',
                        help='Path to (input) IRT Split Directory')

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
    logging.info('    {:24} {}'.format('split-dir', args.rules_dir))
    logging.info('    {:24} {}'.format('model-dir', args.model_dir))
    logging.info('    {:24} {}'.format('--overwrite', args.overwrite))

    logging.info('Environment variables:')
    logging.info('    {:24} {}'.format('PYTHONHASHSEED', os.getenv('PYTHONHASHSEED')))

    return args


def train_ruler(args):
    rules_dir_path = args.rules_dir
    split_dir_path = args.split_dir
    model_dir_path = args.model_dir

    overwrite = args.overwrite

    #
    # Check (input) AnyBURL Rules Directory
    #

    logging.info('Check (input) AnyBURL Rules Directory ...')

    rules_dir = RulesDir(Path(rules_dir_path))
    rules_dir.check()

    #
    # Check (input) IRT Split Directory
    #

    logging.info('Check (input) IRT Split Directory ...')

    split_dir = SplitDir(Path(split_dir_path))
    split_dir.check()

    #
    # Create (output) POWER Model Directory
    #

    logging.info('Create (output) POWER Model Directory ...')

    model_dir = ModelDir(Path(model_dir_path))
    model_dir.create(overwrite=overwrite)

    #
    #
    #

    ent_to_lbl = rules_dir.ent_labels_txt.load()
    rel_to_lbl = rules_dir.rel_labels_txt.load()

    #
    #
    #

    cw_train_triples = split_dir.cw_train_triples_txt.load()
    cw_train_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                      for head, rel, tail in cw_train_triples}

    cw_valid_triples = split_dir.cw_valid_triples_txt.load()
    cw_valid_facts = {Fact.from_ints(head, rel, tail, ent_to_lbl, rel_to_lbl)
                      for head, rel, tail in cw_valid_triples}

    #
    # Read rules
    #

    logging.info('Read rules ...')

    anyburl_rules = rules_dir.cw_train_rules_tsv.load()
    rules = [Rule.from_anyburl(rule, ent_to_lbl, rel_to_lbl) for rule in anyburl_rules]

    good_rules = [rule for rule in rules if rule.conf > 0.5]
    good_rules.sort(key=lambda rule: rule.conf, reverse=True)

    short_rules = [rule for rule in good_rules if len(rule.body) == 1]
    log_rules('Rules', short_rules)

    #
    # Process rules
    #

    logging.info('Process rules ...')

    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', '1234567890'))
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
                ents = [Ent(head.id, ent_to_lbl[head.id]) for head, _, _ in records]

            elif type(body_fact.head) == Ent and type(body_fact.tail) == Var:
                records = session.write_transaction(query_facts_by_head_rel, head=body_fact.head, rel=body_fact.rel)
                ents = [Ent(tail.id, ent_to_lbl[tail.id]) for _, _, tail in records]

            else:
                logging.warning(f'Unsupported rule body in rule {rule}. Skipping.')
                unsupported_rules += 1
                continue

            if logging.getLogger().level == logging.DEBUG:
                groundings = [Fact.from_neo4j(rec) for rec in records]
                log_facts('Groundings', groundings, cw_train_facts, cw_valid_facts)

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

            for fact in pred_facts:
                if fact not in cw_train_facts:
                    pred[fact.head][(fact.rel, fact.tail)].append(rule)

            if logging.getLogger().level == logging.DEBUG:
                log_facts('Predictions', pred_facts, cw_train_facts, cw_valid_facts)

    driver.close()

    #
    # Save ruler
    #

    logging.info('Save ruler ...')

    ruler = Ruler()
    ruler.pred = pred

    model_dir.ruler_pkl.save(ruler)


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


def log_facts(msg: str, facts: List[Fact], cw_train_facts: Set[Fact], cw_valid_facts: Set[Fact], display_max=10):
    if logging.getLogger().level == logging.DEBUG:

        logging.debug(f'{msg} ({display_max}/{len(facts)}):')
        for fact in facts[:display_max]:

            if fact in cw_train_facts:
                split_str = Split.cw_train.name + ':'
            elif fact in cw_valid_facts:
                split_str = Split.cw_valid.name + ':'
            else:
                split_str = str(None) + ':'

            logging.debug(f'{split_str:10} {fact}')


if __name__ == '__main__':
    main()
