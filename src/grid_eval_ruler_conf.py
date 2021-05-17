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

    for min_conf in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        #
        # Variable args
        #

        args.ruler_pkl = f'data/power/ruler-v2/conf/cde-50-test_{min_conf}.pkl'
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
