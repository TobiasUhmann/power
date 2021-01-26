from argparse import ArgumentParser
from os.path import isdir

from pytorch_lightning import Trainer

from ower.classifier import Classifier
from ower.data_module import DataModule


def main():
    #
    # Parse args
    #

    parser = ArgumentParser()

    parser.add_argument('ower_dataset_dir', metavar='ower-dataset-dir',
                        help='Path to (input) OWER Dataset Directory')

    default_gpus = None
    parser.add_argument('--gpus', type=int, metavar='INT', default=default_gpus,
                        help='Train on ... GPUs (default: {})'.format(default_gpus))

    args = parser.parse_args()

    ower_dataset_dir = args.ower_dataset_dir
    gpus = args.gpus

    #
    # Print applied config
    #

    print('Applied config:')
    print('    {:20} {}'.format('ower-dataset-dir', ower_dataset_dir))
    print()
    print('    {:20} {}'.format('--gpus', gpus))
    print()

    #
    # Assert that (input) OWER Dataset Directory exists
    #

    if not isdir(ower_dataset_dir):
        print('OWER Dataset Directory not found')
        exit()

    #
    # Run actual program
    #

    train_classifier(ower_dataset_dir, gpus)


def train_classifier(ower_dataset_dir: str, gpus: int) -> None:
    # Setup DataModule manually to be able to access #classes later
    data_module = DataModule(data_dir=ower_dataset_dir, batch_size=64)
    data_module.prepare_data()

    vocab = data_module.vocab
    num_classes = data_module.num_classes

    classifier = Classifier(vocab_size=len(vocab), embed_dim=32, num_classes=num_classes)

    trainer = Trainer(max_epochs=50, gpus=gpus)
    trainer.fit(classifier, datamodule=data_module)

    trainer.save_checkpoint('data/classifier.ckpt')

    trainer.test(classifier, datamodule=data_module)


if __name__ == '__main__':
    main()
