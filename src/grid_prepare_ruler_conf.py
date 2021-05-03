import logging
from argparse import Namespace

from prepare_ruler import prepare_ruler, log_config


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = Namespace()

    #
    # Fixed args
    #

    args.rules_tsv = f'data/anyburl/fb/rules/rules-1'
    args.url = 'bolt://localhost:7687'
    args.username = 'neo4j'
    args.password = '1234567890'
    args.split_dir = 'data/power/split/fb-30'
    # args.ruler_pkl

    # args.min_conf
    args.min_supp = 1
    args.overwrite = False
    args.random_seed = None

    #
    # Grid Search
    #

    for min_conf in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        #
        # Variable args
        #
        
        # args.rules_tsv
        # args.url
        # args.username
        # args.password
        # args.split_dir
        args.ruler_pkl = f'data/power/ruler-v1/conf/fb-30-test_{min_conf}.pkl'

        args.min_conf = min_conf
        # args.min_supp
        # args.overwrite
        
        #
        # Log config and run
        #

        log_config(args)
        prepare_ruler(args)

    logging.info('Finished successfully')


if __name__ == '__main__':
    main()
