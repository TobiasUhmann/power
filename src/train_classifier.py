import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle
from typing import List, Tuple

import numpy as np
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
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)

    ## Parse args

    args = parse_args()

    ower_dir_path = args.ower_dir
    class_count = args.class_count
    sent_count = args.sent_count

    batch_size = args.batch_size
    device = args.device
    emb_size = args.emb_size
    epoch_count = args.epoch_count
    lr = args.lr
    sent_len = args.sent_len

    ## Check that (input) OWER Directory exists

    ower_dir = OwerDir(Path(ower_dir_path))
    ower_dir.check()

    ## Load datasets

    train_set: List[Sample]
    valid_set: List[Sample]

    train_set, valid_set, _, vocab = ower_dir.read_datasets(class_count, sent_count, vectors='glove.twitter.27B.200d')

    ## Create dataloaders

    def generate_batch(batch: List[Sample]) -> Tuple[Tensor, Tensor, Tensor]:

        ent_batch, gt_classes_batch, tok_lists_batch = zip(*batch)

        cropped_tok_lists_batch = [[tok_list[:sent_len]
                                    for tok_list in tok_lists] for tok_lists in tok_lists_batch]

        padded_tok_lists_batch = [[tok_list + [0] * (sent_len - len(tok_list))
                                   for tok_list in tok_lists] for tok_lists in cropped_tok_lists_batch]

        for padded_tok_lists in padded_tok_lists_batch:
            shuffle(padded_tok_lists)

        return tensor(ent_batch), tensor(padded_tok_lists_batch), tensor(gt_classes_batch)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=generate_batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=generate_batch)

    ## Calc class weights

    _, train_classes_stack, _ = zip(*train_set)
    train_classes_stack = np.array(train_classes_stack)
    train_freqs = train_classes_stack.mean(axis=0)

    class_weights = tensor(1 / train_freqs).to(device)

    ## Create classifier

    classifier = Classifier.from_pre_trained(vocab, class_count).to(device)
    optimizer = Adam(classifier.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss(pos_weight=class_weights)

    writer = SummaryWriter()

    ## Train and validate

    for epoch in range(epoch_count):

        ## Train

        train_loss = 0.0

        # Valid gt/pred classes across all batches
        train_gt_classes_stack: List[List[int]] = []
        train_pred_classes_stack: List[List[int]] = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            _, sents_batch, classes_batch = batch
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

        ## Validate

        valid_loss = 0.0

        # Valid gt/pred classes across all batches
        valid_gt_classes_stack: List[List[int]] = []
        valid_pred_classes_stack: List[List[int]] = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f'Epoch {epoch}'):
                _, sents_batch, classes_batch = batch
                sents_batch = sents_batch.to(device)
                classes_batch = classes_batch.to(device)

                outputs_batch = classifier(sents_batch)
                loss = criterion(outputs_batch, classes_batch.float())

                valid_loss += loss.item()

                pred_classes_batch = (outputs_batch > 0).int()

                valid_gt_classes_stack += classes_batch.cpu().numpy().tolist()
                valid_pred_classes_stack += pred_classes_batch.cpu().numpy().tolist()

        ## Log

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)

        # tps = train precisions, vps = valid precisions, etc.
        tps = precision_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vps = precision_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)
        trs = recall_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vrs = recall_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)
        tfs = f1_score(train_gt_classes_stack, train_pred_classes_stack, average=None)
        vfs = f1_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None)

        for i, (tp, vp, tr, vr, tf, vf) in enumerate(zip(tps, vps, trs, vrs, tfs, vfs)):
            writer.add_scalars('precision', {f'train {i}': tp, f'valid {i}': vp}, epoch)
            writer.add_scalars('recall', {f'train {i}': tr, f'valid {i}': vr}, epoch)
            writer.add_scalars('f1', {f'train {i}': tf, f'valid {i}': vf}, epoch)

        # tps = train precisions, vps = valid precisions, etc.
        tp = precision_score(train_gt_classes_stack, train_pred_classes_stack, average=None).mean()
        vp = precision_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None).mean()
        tr = recall_score(train_gt_classes_stack, train_pred_classes_stack, average=None).mean()
        vr = recall_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None).mean()
        tf = f1_score(train_gt_classes_stack, train_pred_classes_stack, average=None).mean()
        vf = f1_score(valid_gt_classes_stack, valid_pred_classes_stack, average=None).mean()

        writer.add_scalars('precision', {f'train': tp, f'valid': vp}, epoch)
        writer.add_scalars('recall', {f'train': tr, f'valid': vr}, epoch)
        writer.add_scalars('f1', {f'train': tf, f'valid': vf}, epoch)

    ## Save classifier

    with open('data/classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    ## Test classifier

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


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('ower_dir', metavar='ower-dir',
                        help='Path to (input) OWER Directory')

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

    ## Log applied config

    logging.info('Applied config:')
    logging.info('    {:24} {}'.format('ower-dir', args.ower_dir))
    logging.info('    {:24} {}'.format('class-count', args.class_count))
    logging.info('    {:24} {}'.format('sent-count', args.sent_count))
    logging.info('    {:24} {}'.format('--batch-size', args.batch_size))
    logging.info('    {:24} {}'.format('--device', args.device))
    logging.info('    {:24} {}'.format('--emb-size', args.emb_size))
    logging.info('    {:24} {}'.format('--epoch-count', args.epoch_count))
    logging.info('    {:24} {}'.format('--lr', args.lr))
    logging.info('    {:24} {}'.format('--sent-len', args.sent_len))

    return args


if __name__ == '__main__':
    main()
