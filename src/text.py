import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchtext.data import Field, TabularDataset

BATCH_SIZE = 16
EMB_SIZE = 32
NGRAMS = 2


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, emb_size, class_count):
        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, emb_size, sparse=True)
        self.fc = nn.Linear(emb_size, class_count)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)

        return self.fc(embedded)


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


def train_func(sub_train_):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += ((output > 0.5) == cls).sum().item() / 4

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)


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
            acc += ((output > 0.5) == cls).sum().item() / 4

    return loss / len(data_), acc / len(data_)


if __name__ == '__main__':
    if not os.path.isdir('./.data'):
        os.mkdir('./.data')

    #
    # Custom dataset
    #

    tokenize = lambda x: x.split()

    is_male_field = Field(sequential=False, use_vocab=False)
    is_married_field = Field(sequential=False, use_vocab=False)
    is_american_field = Field(sequential=False, use_vocab=False)
    is_actor_field = Field(sequential=False, use_vocab=False)
    context_field = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)

    fields = [('entity', None),
              ('is_male', is_male_field),
              ('is_married', is_married_field),
              ('is_american', is_american_field),
              ('is_actor', is_actor_field),
              ('context', context_field)]

    train_dataset, test_datset = TabularDataset.splits(path='../data',
                                                       train='train_outputs.tsv',
                                                       # validation='valid_outputs.tsv',
                                                       test='test_outputs.tsv',
                                                       format='tsv',
                                                       skip_header=True,
                                                       fields=fields)

    context_field.build_vocab(train_dataset)
    vocab = context_field.vocab

    train_dataset = [(
        [float(x.is_male), float(x.is_married), float(x.is_american), float(x.is_actor)],
        torch.tensor([vocab[t] for t in x.context])
    ) for x in train_dataset]

    test_dataset = [(
        [float(x.is_male), float(x.is_married), float(x.is_american), float(x.is_actor)],
        torch.tensor([vocab[t] for t in x.context])
    ) for x in test_datset]

    # train_iterator, test_iterator = BucketIterator.splits((train_dataset, test_datset),
    #                                                       batch_size=2,
    #                                                       device='cuda')

    #
    #
    #

    # train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    #     root='./.data', ngrams=NGRAMS, vocab=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = len(context_field.vocab)
    nun_class = 4

    # vocab_size = len(train_dataset.get_vocab())
    # nun_class = len(train_dataset.get_labels())

    model = TextSentiment(vocab_size, EMB_SIZE, nun_class).to(device)

    N_EPOCHS = 25
    min_valid_loss = float('inf')

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10., 10., 10., 10.])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

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
