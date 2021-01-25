from typing import List

import torch
from torch import nn
from torchtext.datasets import text_classification, TextClassificationDataset


class TextSentiment(nn.Module):

    embedding: nn.EmbeddingBag
    fc: nn.Linear

    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, num_class)
        self.fc = nn.Linear(embed_dim, num_class)

        self.init_weights()

    def init_weights(self):
        initrange = 0.5

        self.embedding.weight.data.uniform_(-initrange, initrange)

        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.uniform_(-initrange, initrange)

    def forward(self, tokens: List[int], offsets: List[int]):
        embeddings = self.embedding(tokens, offsets)
        outputs = self.fc(embeddings)

        return outputs


def main():
    #
    # Get AG_NEWS dataset
    #

    train_dataset: TextClassificationDataset
    test_dataset: TextClassificationDataset
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root='data/', ngrams=2, vocab=None)

    #
    # Build model
    #

    vocab_size = len(train_dataset.get_vocab())
    embed_dim = 32
    num_classes = len(train_dataset.get_labels())

    text_sentiment = TextSentiment(vocab_size, embed_dim, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    main()
