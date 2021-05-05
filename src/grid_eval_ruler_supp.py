import logging
from argparse import Namespace

from eval_ruler import eval_ruler, log_config


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    args = Namespace()

    #
    # Fixed args
    #

    # args.ruler_pkl
    args.split_dir = 'data/power/split/cde-50'

    args.filter_known = False
    args.test = True

    #
    # Grid Search
    #

    for min_supp in [1, 3, 10, 30, 100, 300, 1000]:
        #
        # Variable args
        #

        args.ruler_pkl = f'data/power/ruler-v2/supp/cde-50-test_{min_supp}.pkl'
        # args.split_dir

        # args.filter_known
        # args.test
        
        #
        # Log config and run
        #

        log_config(args)
        eval_ruler(args)

    logging.info('Finished successfully')


if __name__ == '__main__':
    main()
