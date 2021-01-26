import os
from argparse import ArgumentParser, Namespace
from os import path

import yaml

from ower.classifier import Classifier


def main() -> None:
    #
    # Parse args
    #

    parser = ArgumentParser('Load a trained classifier and run an interactive'
                            ' prompt that lets you query the classifier')

    parser.add_argument('experiment_dir', metavar='experiment-dir',
                        help="Path to (input) experiment directory that contains"
                             " the classifier's checkpoints")

    args = parser.parse_args()

    experiment_dir = args.experiment_dir

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:20} {}'.format('experiment-dir', experiment_dir))
    print()

    #
    # Assert that (input) experiment directory exists
    #

    if not path.isdir(experiment_dir):
        print('Experiment directory not found')
        exit()

    #
    # Run actual program
    #

    run_classifier(experiment_dir)


def run_classifier(experiment_dir: str) -> None:
    classifier = load_classifier_from_experiment(experiment_dir)

    # Put classifier into prediction mode
    classifier.eval()
    classifier.freeze()

    while True:
        print("Enter sentence or 'q' to quit")

        sentence = input('> ')
        if sentence == 'q':
            break

        pred = classifier.predict(sentence)
        print(pred)


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
                                                 hparams=Namespace(**hparams),
                                                 vocab_size=100000,
                                                 embed_dim=32,
                                                 num_class=4)

    return classifier


if __name__ == '__main__':
    main()
