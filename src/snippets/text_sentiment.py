from typing import List

from torch import nn


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
    text_sentiment = TextSentiment(1000, 100, 2)

    return


if __name__ == '__main__':
    main()
