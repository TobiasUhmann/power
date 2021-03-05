import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import Tensor, tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.vocab import Vocab
from tqdm import tqdm

from dao.ower.ower_dir import OwerDir, Sample
from ower.classifier import Classifier


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

    default_batch_size = 1024
    parser.add_argument('--batch-size', dest='batch_size', type=int, metavar='INT', default=default_batch_size,
                        help='Batch size (default: {})'.format(default_batch_size))

    default_emb_size = 256
    parser.add_argument('--emb-size', dest='emb_size', type=int, metavar='INT', default=default_emb_size,
                        help='Embedding size for sentence and class embeddings (default: {})'.format(default_emb_size))

    default_epoch_count = 50
    parser.add_argument('--epoch-count', dest='epoch_count', type=int, metavar='INT', default=default_epoch_count,
                        help='Number of training epochs (default: {})'.format(default_epoch_count))

    default_learning_rate = 0.1
    parser.add_argument('--lr', dest='lr', type=float, metavar='FLOAT', default=default_learning_rate,
                        help='Learning rate (default: {})'.format(default_learning_rate))

    default_sent_len = 64
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

    ower_dir = OwerDir('OWER Directory', Path(args.ower_dataset_dir), class_count, sent_count)
    ower_dir.check()

    #
    # Run actual program
    #

    train_classifier(ower_dir, class_count, sent_count, batch_size, device, emb_size, epoch_count, lr, sent_len)


def train_classifier(ower_dir: OwerDir, class_count: int, sent_count: int, batch_size, device: str, emb_size: int,
                     epoch_count: int, lr: float, sent_len) -> None:

    #
    # Load datasets
    #

    train_set: List[Sample]
    valid_set: List[Sample]

    train_set, valid_set, _, vocab = ower_dir.read_datasets(vectors='glove.twitter.27B.200d')

    #
    # Create dataloaders
    #

    def generate_batch(batch: List[Sample]) -> Tuple[Tensor, Tensor]:

        _, gt_classes_batch, tok_lists_batch = zip(*batch)

        cropped_sents_batch = [[sent[:sent_len]
                                for sent in sents] for sents in tok_lists_batch]

        padded_sents_batch = [[sent + [0] * (sent_len - len(sent))
                               for sent in sents] for sents in cropped_sents_batch]

        return tensor(padded_sents_batch), tensor(gt_classes_batch)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=generate_batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=generate_batch)

    #
    # Create classifier
    #

    # classifier = Classifier.from_random(len(vocab), emb_size, class_count).to(device)
    classifier = Classifier.from_pre_trained(vocab, class_count).to(device)
    print(classifier)
    optimizer = Adam(classifier.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss(pos_weight=tensor([40] * class_count).to(device))

    writer = SummaryWriter()

    #
    # Train and validate
    #

    for epoch in range(epoch_count):

        train_loss = 0.0

        # Valid gt/pred classes across all batches
        train_gt_classes_stack: List[List[int]] = []
        train_pred_classes_stack: List[List[int]] = []

        for batch in tqdm(train_loader, leave=False):
            sents_batch, classes_batch = batch
            sents_batch = sents_batch.to(device)
            classes_batch = classes_batch.to(device)

            outputs_batch = classifier(sents_batch)
            loss = criterion(outputs_batch, classes_batch.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_classes_batch = (outputs_batch > 0).int()

            train_gt_classes_stack += classes_batch.cpu().numpy().tolist()
            train_pred_classes_stack += pred_classes_batch.cpu().numpy().tolist()


            train_loss += loss.item()

        valid_loss = 0.0

        # Valid gt/pred classes across all batches
        valid_gt_classes_stack: List[List[int]] = []
        valid_pred_classes_stack: List[List[int]] = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, leave=False):
                sents_batch, classes_batch = batch
                sents_batch = sents_batch.to(device)
                classes_batch = classes_batch.to(device)

                outputs_batch = classifier(sents_batch)
                loss = criterion(outputs_batch, classes_batch.float())

                valid_loss += loss.item()

                pred_classes_batch = (outputs_batch > 0).int()

                valid_gt_classes_stack += classes_batch.cpu().numpy().tolist()
                valid_pred_classes_stack += pred_classes_batch.cpu().numpy().tolist()


        std_train_loss = train_loss / len(train_loader)
        std_valid_loss = valid_loss / len(valid_loader)

        logging.info('Epoch {}: train loss = {:.2e}, valid loss = {:.2e}'.format(
            epoch, std_train_loss, std_valid_loss))

        writer.add_scalars('loss', {'train': std_train_loss, 'valid': std_valid_loss}, epoch)

        # tps = train precisions, vps = valid precisions, etc.
        tps = precision_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vps = precision_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)
        trs = recall_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vrs = recall_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)
        tfs = f1_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vfs = f1_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)

        mean_train_precision = tps.mean()
        mean_valid_precision = vps.mean()
        mean_train_recall = trs.mean()
        mean_valid_recall = vrs.mean()
        mean_train_f1 = tfs.mean()
        mean_valid_f1 = vfs.mean()

        print(f'    Precision:  train = {mean_train_precision:.2f}, valid = {mean_valid_precision:.2f}')
        print(f'    Recall:  train = {mean_train_recall:.2f}, valid = {mean_valid_recall:.2f}')
        print(f'    F1:  train = {mean_train_f1:.2f}, valid = {mean_valid_f1:.2f}')
        writer.add_scalars('precision', {f'train': mean_train_precision, f'valid': mean_valid_precision}, epoch)
        writer.add_scalars('recall', {f'train': mean_train_recall, f'valid': mean_valid_recall}, epoch)
        writer.add_scalars('f1', {f'train': mean_train_f1, f'valid': mean_valid_f1}, epoch)

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

    pred_classes = predict(ex_text_strs, classifier, vocab)
    # pred_labels = [class_labels[pred_class] for pred_class in pred_classes if pred_class == 1]

    logging.info('Barack Obama')
    for i, pred_class in enumerate(pred_classes[0]):
        logging.info('{}: {}'.format(class_labels[i], pred_class))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
    main()
