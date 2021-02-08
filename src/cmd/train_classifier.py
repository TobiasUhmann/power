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
                             num_classes=4, num_sentences=3, sent_len=64)
    data_module.prepare_data()

    vocab = data_module.vocab
    num_classes = data_module.num_classes

    classifier = Classifier(vocab_size=len(vocab), emb_size=32,
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

        tokens_1 = [vocab[word] for word in words_1]
        tokens_2 = [vocab[word] for word in words_2]
        tokens_3 = [vocab[word] for word in words_3]

        with torch.no_grad():
            sents = torch.tensor([tokens_1 + [0] * (64 - len(tokens_1)),
                                  tokens_2 + [0] * (64 - len(tokens_2)),
                                  tokens_3 + [0] * (64 - len(tokens_3))
                                  ]).unsqueeze(0).cuda()

            class_logits: Tensor = classifier(sents)
            pred_classes = class_logits > 0.5

            return pred_classes

    ex_text_str_1 = "Barack Obama is a married american actor."
    ex_text_str_2 = "Barack Obama is a married american actor."
    ex_text_str_3 = "Barack Obama is a married american actor."

    ex_text_strs = [ex_text_str_1, ex_text_str_2, ex_text_str_3]

    # classifier = classifier.to('cpu')
    vocab = data_module.vocab

    pred_classes = predict(ex_text_strs, classifier, vocab)
    # pred_labels = [class_labels[pred_class] for pred_class in pred_classes if pred_class == 1]

    print()
    print('Barack Obama: ', pred_classes)


if __name__ == '__main__':
    main()
