#
# Refactored version of torchtext_classifier.py
#

import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torchtext
from torch import Tensor, optim
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import random_split
from torchtext.data.utils import ngrams_iterator
from torchtext.datasets import text_classification
from torchtext.vocab import Vocab

BATCH_SIZE = 16
EMBED_DIM = 32
NGRAMS = 2
NUM_EPOCHS = 5


#
# Define the model
#

class TextSentiment(nn.Module):
    embedding: nn.EmbeddingBag
    fc: nn.Linear

    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5

        self.embedding.weight.data.uniform_(-initrange, initrange)

        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, concated_token_lists: List[int], offsets: List[int]) -> Tensor:
        """
        :return: Shape [batch_size][num_classes]
        """

        # Shape [batch_size][embed_dim]
        embeddings: Tensor = self.embedding(concated_token_lists, offsets)

        # Shape [batch_size][num_classes]
        outputs: Tensor = self.fc(embeddings)

        return outputs


def main():
    #
    # Load data with ngrams
    #

    if not os.path.isdir('data/'):
        os.mkdir('data/')

    ag_news = text_classification.DATASETS['AG_NEWS']
    train_valid_dataset, test_dataset = ag_news(root='data/', ngrams=NGRAMS, vocab=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #
    # Instantiate the instance
    #

    vocab_size = len(train_valid_dataset.get_vocab())
    num_classes = len(train_valid_dataset.get_labels())

    model = TextSentiment(vocab_size, EMBED_DIM, num_classes).to(device)

    #
    # Split the dataset and run the model
    #

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=4.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(train_valid_dataset) * 0.95)
    valid_len = len(train_valid_dataset) - train_len
    train_dataset, valid_dataset = random_split(train_valid_dataset, [train_len, valid_len])

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_func(device, model, criterion, optimizer, scheduler, train_dataset)
        valid_loss, valid_acc = test(device, model, criterion, valid_dataset)

        total_secs = int(time.time() - start_time)
        mins = int(total_secs / 60)
        secs = total_secs % 60

        print()
        print(f'Epoch: {epoch + 1} | time in {mins} minutes, {secs} seconds')
        print(f'\tTrain | loss = {train_loss:.4f}\t|\tacc = {train_acc * 100:.1f}%')
        print(f'\tValid | loss = {valid_loss:.4f}\t|\tacc = {valid_acc * 100:.1f}%')

    #
    # Evaluate the model with test dataset
    #

    test_loss, test_acc = test(device, model, criterion, test_dataset)

    print()
    print('Checking the results of test dataset ...')
    print(f'\tTest  | loss = {test_loss:.4f}\t|\tacc = {test_acc * 100:.1f}%')

    #
    # Test on a random news
    #

    class_to_label = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tec'}

    def predict(text: str, model: nn.Module, vocab: Vocab, ngrams: int):
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        words: List[str] = tokenizer(text)

        with torch.no_grad():
            tokens = torch.tensor([vocab[token] for token in ngrams_iterator(words, ngrams)])

            class_logits: Tensor = model(tokens, torch.tensor([0]))
            pred_class = class_logits.argmax(1).item() + 1

            return pred_class

    ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was enduring the season’s" \
                  " worst weather conditions on Sunday at The Open on his way to a" \
                  " closing 75 at Royal Portrush, which considering the wind and the" \
                  " rain was a respectable showing. Thursday’s first round at the" \
                  " WGC-FedEx St. Jude Invitational was another story. With temperatures" \
                  " in the mid-80s and hardly any wind, the Spaniard was 13 strokes" \
                  " better in a flawless round. Thanks to his best putting performance" \
                  " on the PGA Tour, Rahm     finished with an 8-under 62 for a" \
                  " three-stroke lead, which was even more impressive considering he’d" \
                  " never played the front nine at TPC Southwind."

    model = model.to('cpu')
    vocab = train_valid_dataset.get_vocab()

    pred_class = predict(ex_text_str, model, vocab, ngrams=2)
    pred_label = class_to_label[pred_class]

    print()
    print(f'This is a {pred_label} news')


def generate_batch(
        label_tokens_batch: List[Tuple[int, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Split (label, tokens) batch and transform tokens into EmbeddingBag format.

    :return: 1. Concated tokens of all texts, Tensor[]
             2. Token offsets where texts begin, Tensor[batch_size]
             3. Labels for texts, Tensor[batch_size]
    """

    label_batch = torch.tensor([entry[0] for entry in label_tokens_batch])
    tokens_batch = [entry[1] for entry in label_tokens_batch]

    num_tokens_batch = [len(tokens) for tokens in tokens_batch]

    offset_batch = torch.tensor([0] + num_tokens_batch[:-1]).cumsum(dim=0)
    concated_tokens_batch = torch.cat(tokens_batch)

    return concated_tokens_batch, offset_batch, label_batch


def train_func(
        device: Device,
        model: nn.Module,
        criterion,
        optimizer: Optimizer,
        scheduler,
        dataset: Dataset
) -> Tuple[float, float]:
    """
    :return: 1. Epoch loss
             2. Epoch accuracy
    """

    epoch_loss_sum: float = 0
    epoch_acc_sum: float = 0

    data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    for concated_tokens_batch, offset_batch, label_batch, in data:
        concated_tokens_batch = concated_tokens_batch.to(device)
        offset_batch = offset_batch.to(device)
        label_batch = label_batch.to(device)

        # Shape [batch_size][num_classes]
        output_batch = model(concated_tokens_batch, offset_batch)

        loss = criterion(output_batch, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_sum += loss.item()
        epoch_acc_sum += (output_batch.argmax(1) == label_batch).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return epoch_loss_sum / len(dataset), epoch_acc_sum / len(dataset)


def test(
        device: Device,
        model: nn.Module,
        criterion,
        dataset: Dataset
) -> Tuple[float, float]:
    """
    :return: 1. Epoch loss
             2. Epoch accuracy
    """

    epoch_loss_sum = 0
    epoch_acc_sum = 0

    data = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=generate_batch)

    for concated_tokens_batch, offset_batch, label_batch in data:
        concated_tokens_batch = concated_tokens_batch.to(device)
        offset_batch = offset_batch.to(device)
        label_batch = label_batch.to(device)

        with torch.no_grad():
            output_batch = model(concated_tokens_batch, offset_batch)

            loss = criterion(output_batch, label_batch)

            epoch_loss_sum += loss.item()
            epoch_acc_sum += (output_batch.argmax(1) == label_batch).sum().item()

    return epoch_loss_sum / len(dataset), epoch_acc_sum / len(dataset)


if __name__ == '__main__':
    main()
