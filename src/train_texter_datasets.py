import logging
from argparse import Namespace

from train_texter import train_texter


def main():
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    #
    # Specify config
    #

    args = Namespace()

    # args.samples_dir
    args.class_count = 100
    # args.sent_count
    # args.split_dir
    # args.texter_pkl

    # args.batch_size
    args.device = 'cuda'
    args.epoch_count = 20
    # args.log_dir
    args.log_steps = False
    args.lr = 1e-5
    # args.overwrite
    args.sent_len = 64
    # args.try_batch_size

    #
    # Combinations
    #

    a = 'data/power/samples'
    b = 'data/power/split'
    c = 'data/power/texter'

    # [(samples_dir, sent_count, split_dir, texter_pkl, batch_size, log_dir)]
    # Start out with twice the batch size that works on a GTX 1080 Ti with 11GB RAM.
    # [[(dataset, sent count), BASE batch size, OWER batch size]]
    combos = [
        [f'{a}/cde-cde-1-clean/', 1, f'{b}/cde-0/', f'{c}/cde-cde-1-clean.pkl', 256, 'runs/cde-cde-1-clean/'],
        [f'{a}/cde-cde-5-clean/', 5, f'{b}/cde-0/', f'{c}/cde-cde-5-clean.pkl', 64, 'runs/cde-cde-5-clean/'],
        [f'{a}/cde-cde-15-clean/', 15, f'{b}/cde-0/', f'{c}/cde-cde-15-clean.pkl', 16, 'runs/cde-cde-15-clean/'],
        [f'{a}/cde-cde-30-clean/', 30, f'{b}/cde-0/', f'{c}/cde-cde-30-clean.pkl', 8, 'runs/cde-cde-30-clean/'],
        [f'{a}/cde-irt-1-clean/', 1, f'{b}/cde-0/', f'{c}/cde-irt-1-clean.pkl', 256, 'runs/cde-irt-1-clean/'],
        [f'{a}/cde-irt-1-marked/', 1, f'{b}/cde-0/', f'{c}/cde-irt-1-marked.pkl', 256, 'runs/cde-irt-1-marked/'],
        [f'{a}/cde-irt-1-masked/', 1, f'{b}/cde-0/', f'{c}/cde-irt-1-masked.pkl', 256, 'runs/cde-irt-1-masked/'],
        [f'{a}/cde-irt-5-clean/', 5, f'{b}/cde-0/', f'{c}/cde-irt-5-clean.pkl', 64, 'runs/cde-irt-5-clean/'],
        [f'{a}/cde-irt-5-marked/', 5, f'{b}/cde-0/', f'{c}/cde-irt-5-marked.pkl', 64, 'runs/cde-irt-5-marked/'],
        [f'{a}/cde-irt-5-masked/', 5, f'{b}/cde-0/', f'{c}/cde-irt-5-masked.pkl', 64, 'runs/cde-irt-5-masked/'],
        [f'{a}/cde-irt-15-clean/', 15, f'{b}/cde-0/', f'{c}/cde-irt-15-clean.pkl', 16, 'runs/cde-irt-15-clean/'],
        [f'{a}/cde-irt-15-marked/', 15, f'{b}/cde-0/', f'{c}/cde-irt-15-marked.pkl', 16, 'runs/cde-irt-15-marked/'],
        [f'{a}/cde-irt-15-masked/', 15, f'{b}/cde-0/', f'{c}/cde-irt-15-masked.pkl', 16, 'runs/cde-irt-15-masked/'],
        [f'{a}/cde-irt-30-clean/', 30, f'{b}/cde-0/', f'{c}/cde-irt-30-clean.pkl', 8, 'runs/cde-irt-30-clean/'],
        [f'{a}/cde-irt-30-marked/', 30, f'{b}/cde-0/', f'{c}/cde-irt-30-marked.pkl', 8, 'runs/cde-irt-30-marked/'],
        [f'{a}/cde-irt-30-masked/', 30, f'{b}/cde-0/', f'{c}/cde-irt-30-masked.pkl', 8, 'runs/cde-irt-30-masked/'],
        [f'{a}/fb-irt-1-clean/', 1, f'{b}/fb-0/', f'{c}/fb-irt-1-clean.pkl', 256, 'runs/fb-irt-1-clean/'],
        [f'{a}/fb-irt-1-marked/', 1, f'{b}/fb-0/', f'{c}/fb-irt-1-marked.pkl', 256, 'runs/fb-irt-1-marked/'],
        [f'{a}/fb-irt-1-masked/', 1, f'{b}/fb-0/', f'{c}/fb-irt-1-masked.pkl', 256, 'runs/fb-irt-1-masked/'],
        [f'{a}/fb-irt-5-clean/', 5, f'{b}/fb-0/', f'{c}/fb-irt-5-clean.pkl', 64, 'runs/fb-irt-5-clean/'],
        [f'{a}/fb-irt-5-marked/', 5, f'{b}/fb-0/', f'{c}/fb-irt-5-marked.pkl', 64, 'runs/fb-irt-5-marked/'],
        [f'{a}/fb-irt-5-masked/', 5, f'{b}/fb-0/', f'{c}/fb-irt-5-masked.pkl', 64, 'runs/fb-irt-5-masked/'],
        [f'{a}/fb-irt-15-clean/', 15, f'{b}/fb-0/', f'{c}/fb-irt-15-clean.pkl', 16, 'runs/fb-irt-15-clean/'],
        [f'{a}/fb-irt-15-marked/', 15, f'{b}/fb-0/', f'{c}/fb-irt-15-marked.pkl', 16, 'runs/fb-irt-15-marked/'],
        [f'{a}/fb-irt-15-masked/', 15, f'{b}/fb-0/', f'{c}/fb-irt-15-masked.pkl', 16, 'runs/fb-irt-15-masked/'],
        [f'{a}/fb-irt-30-clean/', 30, f'{b}/fb-0/', f'{c}/fb-irt-30-clean.pkl', 8, 'runs/fb-irt-30-clean/'],
        [f'{a}/fb-irt-30-marked/', 30, f'{b}/fb-0/', f'{c}/fb-irt-30-marked.pkl', 8, 'runs/fb-irt-30-marked/'],
        [f'{a}/fb-irt-30-masked/', 30, f'{b}/fb-0/', f'{c}/fb-irt-30-masked.pkl', 8, 'runs/fb-irt-30-masked/'],
        [f'{a}/fb-owe-1-clean/', 1, f'{b}/fb-0/', f'{c}/fb-owe-1-clean.pkl', 256, 'runs/fb-owe-1-clean/'],
    ]

    #
    # Try batches sizes. Decrease if graphics RAM is not sufficient until it fits.
    #

    for samples_dir, sent_count, split_dir, _, batch_size, _ in combos:

        args.samples_dir = samples_dir
        args.sent_count = sent_count
        args.texter_pkl = 'data/power/texter/try_batch_size.pkl'
        args.split_dir = split_dir

        args.batch_size = batch_size
        args.log_dir = 'runs/try_batch_size/'
        args.overwrite = True
        args.try_batch_size = True

        while True:
            try:
                logging.info(f'Try batch size {batch_size} for samples {samples_dir}')
                train_texter(args)

                # Halve once more, just to be safe
                batch_size //= 2
                combos[-2] = batch_size

                logging.info(f'Works. Use batch size {batch_size} for samples {samples_dir}')
                break

            except RuntimeError:
                logging.warning(f'Batch size {batch_size} too large for samples {samples_dir}.'
                                f' Halve batch size to {batch_size // 2}.')

                batch_size //= 2
                combos[-2] = batch_size
                args.batch_size = batch_size

    #
    # Perform grid search
    #

    for i in range(3):
        for samples_dir, sent_count, split_dir, texter_pkl, batch_size, log_dir in combos:
            args.samples_dir = samples_dir
            args.sent_count = sent_count
            args.split_dir = split_dir
            args.texter_pkl = texter_pkl

            args.batch_size = batch_size
            args.log_dir = log_dir
            args.overwrite = False
            args.try_batch_size = False

            logging.info(f'Training on samples {samples_dir}')
            train_texter(args)


if __name__ == '__main__':
    main()
