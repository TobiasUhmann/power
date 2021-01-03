from argparse import ArgumentParser

from pytorch_lightning import Trainer

from power.classifier import Classifier
from power.data_module import DataModule


def main():
    #
    # Parse args
    #

    parser = ArgumentParser()

    default_gpus = None
    parser.add_argument('--gpus', type=int, metavar='INT', default=default_gpus,
                        help='Train on ... GPUs (default: {})'.format(default_gpus))

    args = parser.parse_args()

    gpus = args.gpus

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:20} {}'.format('--gpus', gpus))
    print()

    #
    # Train classifier
    #

    data_module = DataModule(data_dir='data/', batch_size=64)

    classifier = Classifier(vocab_size=100000,
                            embed_dim=32,
                            num_class=4)

    if gpus:
        trainer = Trainer(gpus=gpus)
    else:
        trainer = Trainer()

    trainer.fit(classifier, data_module)


if __name__ == '__main__':
    main()
