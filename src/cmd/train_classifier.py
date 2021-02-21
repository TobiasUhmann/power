import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchtext.vocab import Vocab
from tqdm import tqdm

from dao.ower.ower_dir import OwerDir
from ower.classifier import Classifier
from ower.data_module import DataModule


def main():
    #
    # Parse args
    #

    parser = ArgumentParser()

    parser.add_argument('ower_dataset_dir', metavar='ower-dataset-dir',
                        help='Path to (input) OWER Dataset Directory')

    parser.add_argument('class_count', metavar='class-count', type=int,
                        help='Number of classes distinguished by the classifier')

    parser.add_argument('sent_count', metavar='sent-count', type=int,
                        help='Number of sentences per entity')

    device_choices = ['cpu', 'cuda']
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', metavar='STR', choices=device_choices, default=default_device,
                        help='Where to perform tensor operations, one of {} (default: {})'.format(
                            device_choices, default_device))

    default_batch_size = 64
    parser.add_argument('--batch-size', dest='batch_size', type=int, metavar='INT', default=default_batch_size,
                        help='Batch size (default: {})'.format(default_batch_size))

    default_emb_size = 32
    parser.add_argument('--emb-size', dest='emb_size', type=int, metavar='INT', default=default_emb_size,
                        help='Embedding size for sentence and class embeddings (default: {})'.format(default_emb_size))

    default_epoch_count = 10
    parser.add_argument('--epoch-count', dest='epoch_count', type=int, metavar='INT', default=default_epoch_count,
                        help='Number of training epochs (default: {})'.format(default_epoch_count))

    default_learning_rate = 0.01
    parser.add_argument('--lr', dest='lr', type=float, metavar='FLOAT', default=default_learning_rate,
                        help='Learning rate (default: {})'.format(default_learning_rate))

    default_sent_len = 32
    parser.add_argument('--sent-len', dest='sent_len', type=int, metavar='INT', default=default_sent_len,
                        help='Sentence length short sentences are padded and long sentences cropped to'
                             ' (default: {})'.format(default_sent_len))

    args = parser.parse_args()

    ower_dataset_dir = args.ower_dataset_dir
    class_count = args.class_count
    sent_count = args.sent_count

    batch_size = args.batch_size
    device = args.device
    emb_size = args.emb_size
    epoch_count = args.epoch_count
    lr = args.lr
    sent_len = args.sent_len

    #
    # Print applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ower-dataset-dir', ower_dataset_dir))
    logging.info('    {:24} {}'.format('class-count', class_count))
    logging.info('    {:24} {}'.format('sent-count', sent_count))
    logging.info('')
    logging.info('    {:24} {}'.format('--batch-size', batch_size))
    logging.info('    {:24} {}'.format('--device', device))
    logging.info('    {:24} {}'.format('--emb-size', emb_size))
    logging.info('    {:24} {}'.format('--epoch-count', epoch_count))
    logging.info('    {:24} {}'.format('--lr', lr))
    logging.info('    {:24} {}'.format('--sent-len', sent_len))
    logging.info('')

    #
    # Check that OWER Directory exists
    #

    ower_dir = OwerDir('OWER Directory', Path(args.ower_dataset_dir))
    ower_dir.check()

    #
    # Run actual program
    #

    train_classifier(ower_dir, class_count, sent_count, batch_size, device, emb_size, epoch_count, lr, sent_len)


def train_classifier(ower_dir: OwerDir, class_count: int, sent_count: int, batch_size, device: str, emb_size: int,
                     epoch_count: int, lr: float, sent_len) -> None:

    #
    # Load data
    #

    data_module = DataModule(str(ower_dir._path), class_count, sent_count, batch_size, sent_len)
    data_module.load_datasets()

    train_loader = data_module.get_train_loader()
    valid_loader = data_module.get_valid_loader()

    #
    # Create classifier
    #

    classifier = Classifier(vocab_size=len(data_module.vocab), emb_size=emb_size, class_count=class_count).to(device)
    optimizer = Adam(classifier.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([80] * class_count).to(device))

    writer = SummaryWriter()

    #
    # Train and validate
    #

    for epoch in range(epoch_count):

        train_loss = 0.0
        for batch in tqdm(train_loader, leave=False):
            sents_batch, classes_batch = batch
            sents_batch = sents_batch.to(device)
            classes_batch = classes_batch.to(device)

            outputs_batch = classifier(sents_batch)
            loss = criterion(outputs_batch, classes_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        valid_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, leave=False):
                sents_batch, classes_batch = batch
                sents_batch = sents_batch.to(device)
                classes_batch = classes_batch.to(device)

                outputs_batch = classifier(sents_batch)
                loss = criterion(outputs_batch, classes_batch.float())

                valid_loss += loss.item()

        std_train_loss = train_loss / len(train_loader)
        std_valid_loss = valid_loss / len(valid_loader)

        logging.info('Epoch {}: train loss = {:.2e}, valid loss = {:.2e}'.format(
            epoch, std_train_loss, std_valid_loss))

        writer.add_scalars('loss', {'train': std_train_loss, 'valid': std_valid_loss}, epoch)

    #
    # Save classifier
    #

    with open('data/classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    #
    # Test classifier
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
            sents = torch.tensor([tokens_1 + [0] * (sent_len - len(tokens_1)),
                                  tokens_2 + [0] * (sent_len - len(tokens_2)),
                                  tokens_3 + [0] * (sent_len - len(tokens_3))
                                  ]).unsqueeze(0).to(device)

            class_logits: Tensor = classifier(sents)
            pred_classes = class_logits > 0.5

            logging.info(f'class_logits = {class_logits}')

            return pred_classes

    ex_text_str_1 = "Barack Obama is married"
    ex_text_str_2 = "Barack Obama is American"
    ex_text_str_3 = "Barack Obama is an actor"

    ex_text_strs = [ex_text_str_1, ex_text_str_2, ex_text_str_3]

    pred_classes = predict(ex_text_strs, classifier, data_module.vocab)
    # pred_labels = [class_labels[pred_class] for pred_class in pred_classes if pred_class == 1]

    logging.info('Barack Obama')
    for i, pred_class in enumerate(pred_classes[0]):
        logging.info('{}: {}'.format(class_labels[i], pred_class))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
    main()
