from argparse import ArgumentParser
from os.path import isdir
from typing import List

import torch
from pytorch_lightning import Trainer
from torch import Tensor
from torchtext.vocab import Vocab

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
    data_module = DataModule(data_dir=ower_dataset_dir, batch_size=64,
                             num_classes=4, num_sentences=3)
    data_module.prepare_data()

    vocab = data_module.vocab
    num_classes = data_module.num_classes

    classifier = Classifier(vocab_size=len(vocab), embed_dim=32,
                            num_classes=num_classes)

    trainer = Trainer(max_epochs=5, gpus=gpus)
    trainer.fit(classifier, datamodule=data_module)

    trainer.save_checkpoint('data/classifier.ckpt')

    trainer.test(classifier, datamodule=data_module)

    #
    # Predict custom sample
    #

    class_labels = ['is_married', 'is_male', 'is_american', 'is_actor']

    def predict(texts: List[str], classifier: Classifier, vocab: Vocab):
        text_1, text_2, text_3 = texts
        words_1 = text_1.split()
        words_2 = text_2.split()
        words_3 = text_3.split()

        with torch.no_grad():
            tokens_1 = torch.tensor([vocab[word] for word in words_1])
            tokens_2 = torch.tensor([vocab[word] for word in words_2])
            tokens_3 = torch.tensor([vocab[word] for word in words_3])

            class_logits: Tensor = classifier(
                tokens_1, torch.tensor([0]),
                tokens_2, torch.tensor([0]),
                tokens_3, torch.tensor([0]),
            )
            pred_classes = class_logits > 0.5

            return pred_classes

    ex_text_str_1 = "American actress."
    ex_text_str_2 = "American actress."
    ex_text_str_3 = "American actress."

    ex_text_strs = [ex_text_str_1, ex_text_str_2, ex_text_str_3]

    classifier = classifier.to('cpu')
    vocab = data_module.vocab

    pred_classes = predict(ex_text_strs, classifier, vocab)
    # pred_labels = [class_labels[pred_class] for pred_class in pred_classes if pred_class == 1]

    print()
    print('Kamala Harris: ', pred_classes)


if __name__ == '__main__':
    main()
