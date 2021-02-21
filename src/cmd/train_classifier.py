import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
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

    #
    # Print applied config
    #

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ower-dataset-dir', args.ower_dataset_dir))
    logging.info('    {:24} {}'.format('class-count', args.class_count))
    logging.info('    {:24} {}'.format('sent-count', args.sent_count))
    logging.info('')
    logging.info('    {:24} {}'.format('--batch-size', args.batch_size))
    logging.info('    {:24} {}'.format('--device', args.device))
    logging.info('    {:24} {}'.format('--emb-size', args.emb_size))
    logging.info('    {:24} {}'.format('--epoch-count', args.epoch_count))
    logging.info('    {:24} {}'.format('--lr', args.lr))
    logging.info('    {:24} {}'.format('--sent-len', args.sent_len))
    logging.info('')

    #
    # Check that OWER Directory exists
    #

    ower_dir = OwerDir('OWER Directory', Path(args.ower_dataset_dir))
    ower_dir.check()

    #
    # Run actual program
    #

    train_classifier(ower_dir, args.class_count, args.sent_count, args.batch_size, args.device, args.emb_size,
                     args.epoch_count, args.lr, args.sent_len)


def train_classifier(ower_dir: OwerDir, class_count: int, sent_count: int, batch_size, device: str, emb_size: int,
                     epoch_count: int, lr: float, sent_len) -> None:
    data_module = DataModule(str(ower_dir._path), class_count, sent_count, batch_size, sent_len)
    data_module.load_datasets()

    train_loader = data_module.get_train_loader()
    valid_loader = data_module.get_valid_loader()

    classifier = Classifier(vocab_size=len(data_module.vocab), emb_size=emb_size, class_count=class_count).to(device)
    optimizer = Adam(classifier.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([10] * class_count).to(device))

    writer = SummaryWriter()

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


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
    main()
