from typing import List

from pytorch_lightning import LightningModule
from torch import optim, Tensor
from torch.nn import EmbeddingBag, Linear


class Classifier(LightningModule):

    embedding: EmbeddingBag
    fc: Linear

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()

        # Create layers
        self.embedding = EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = Linear(embed_dim, num_classes)

        # Init weights
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=4.0)

    def forward(self, tokens_batch_concated: List[int], offsets: List[int]):
        """
        :return: Class logits, shape [batch_size][class_count]
        """

        # Shape [batch_size][embed_dim]
        embeddings: Tensor = self.embedding(tokens_batch_concated, offsets)

        # Shape [batch_size][class_count]
        class_logits: Tensor = self.fc(embeddings)

        return class_logits
