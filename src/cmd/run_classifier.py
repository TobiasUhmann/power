import os
from argparse import ArgumentParser, Namespace
from os import path

import yaml

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

    if not path.isdir(experiment):
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


def load_classifier_from_experiment(experiment_dir: str) -> Classifier:
    """
    Load the classifier from the latest checkpoint in an experiment directory.
    """

    hparams_yaml = path.join(experiment_dir, 'hparams.yaml')
    with open(hparams_yaml, encoding='utf-8') as f:
        hparams = yaml.load(f.read(), Loader=yaml.FullLoader)

    checkpoints_dir = path.join(experiment_dir, 'checkpoints')
    checkpoints = [
        file
        for file in os.listdir(checkpoints_dir)
        if file.endswith('.ckpt')
    ]

    latest_classifier_ckpt = path.join(checkpoints_dir, checkpoints[-1])

    classifier = Classifier.load_from_checkpoint(latest_classifier_ckpt,
                                                 hparams=Namespace(**hparams))

    return classifier


if __name__ == '__main__':
    main()
