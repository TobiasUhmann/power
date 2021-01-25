#
# Tidied up version of torchtext_classifier_improved.py
#

import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchtext.datasets import text_classification

BATCH_SIZE = 16
EMBED_DIM = 32
NGRAMS = 2

#
# Load data with ngrams
#

if not os.path.isdir('data/'):
    os.mkdir('data/')

ag_news = text_classification.DATASETS['AG_NEWS']
train_dataset, test_dataset = ag_news(root='data/', ngrams=NGRAMS, vocab=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        :return: Shape [batch_size][class_count]
        """

        # Shape [batch_size][embed_dim]
        embeddings: Tensor = self.embedding(concated_token_lists, offsets)

        # Shape [batch_size][class_count]
        outputs: Tensor = self.fc(embeddings)

        return outputs


#
# Instantiate the instance
#

vocab_size = len(train_dataset.get_vocab())
class_count = len(train_dataset.get_labels())

model = TextSentiment(vocab_size, EMBED_DIM, class_count).to(device)


#
# Functions used to generate batch
#

def generate_batch(label_tokens_batch: List[Tuple[int, Tensor]]) \
        -> Tuple[Tensor, Tensor, Tensor]:
    """
    Split (label, tokens) batch and transform tokens into EmbeddingBag format.

    :return: 1. Concated tokens of all texts, Tensor[]
             2. Token offsets where texts begin, Tensor[batch_size]
             3. Labels for texts, Tensor[batch_size]
    """

    label_batch = torch.tensor([entry[0] for entry in label_tokens_batch])
    tokens_batch = [entry[1] for entry in label_tokens_batch]

    token_count_batch = [len(tokens) for tokens in tokens_batch]

    offset_batch = torch.tensor([0] + token_count_batch[:-1]).cumsum(dim=0)
    concated_tokens_batch = torch.cat(tokens_batch)

    return concated_tokens_batch, offset_batch, label_batch


#
# Define functions to train the model and evaluate results
#

def train_func(dataset: Dataset) -> Tuple[float, float]:
    """
    :return: 1. Epoch loss
             2. Epoch accuracy
    """

    epoch_loss: float = 0
    epoch_acc: float = 0

    data = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    for concated_tokens_batch, offset_batch, label_batch, in data:
        concated_tokens_batch = concated_tokens_batch.to(device)
        offset_batch = offset_batch.to(device)
        label_batch = label_batch.to(device)

        # Shape [batch_size][class_count]
        output_batch = model(concated_tokens_batch, offset_batch)

        loss = criterion(output_batch, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += (output_batch.argmax(1) == label_batch).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return epoch_loss / len(dataset), epoch_acc / len(dataset)


def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


#
# Split the dataset and run the model
#

import time
from torch.utils.data.dataset import random_split

N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

#
# Evaluate the model with test dataset
#

print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

#
# Test on a random news
#

from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}


def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                             for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" % ag_news_label[predict(ex_text_str, model, vocab, 2)])
