from argparse import ArgumentParser
from os.path import isdir

from ower.classifier import Classifier
from ower.data_module import DataModule


def main():
    #
    # Parse args
    #

    parser = ArgumentParser('Load a trained classifier and run an interactive'
                            ' prompt that lets you query the classifier')

    parser.add_argument('experiment',
                        help="Path to (input) experiment directory that contains"
                             " the classifier's checkpoints")

    default_gpus = None
    parser.add_argument('--gpus', type=int, metavar='INT', default=default_gpus,
                        help='Train on ... GPUs (default: {})'.format(default_gpus))

    args = parser.parse_args()

    experiment = args.experiment
    gpus = args.gpus

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:20} {}'.format('experiment', experiment))
    print()
    print('    {:20} {}'.format('--gpus', gpus))
    print()

    #
    # Assert that (input) experiment directory exists
    #

    if not isdir(experiment):
        print('Experiment directory not found')
        exit()

    #
    # Run actual program
    #

    train_classifier(experiment, gpus)


def train_classifier(experiment: str, gpus: int) -> None:
    # Setup DataModule manually to be able to access #classes later
    dm = DataModule(data_dir=experiment, batch_size=64)
    dm.prepare_data()
    dm.setup('fit')

    classifier = Classifier.load_from_checkpoint('data/ower.ckpt')
    classifier.eval()

    valid_loader = dm.val_dataloader()
    sample = valid_loader.dataset[0]


if __name__ == '__main__':
    main()
