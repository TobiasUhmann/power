from typing import List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from torch import Tensor
from torch.nn import EmbeddingBag, Linear, CrossEntropyLoss


class Classifier(LightningModule):

    embedding: EmbeddingBag
    fc: Linear

    acc: Accuracy

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super(Classifier, self).__init__()

        # Create layers
        self.embedding = EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = Linear(embed_dim, num_classes)

        # Init weights
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

        # Add metrics
        self.acc = pl.metrics.Accuracy()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=4.0)

    def forward(self, tokens_batch_concated: List[int], offsets: List[int]):
        """
        :return: Class logits, shape [batch_size][class_count]
        """

        # Shape [batch_size][embed_dim]
        embeddings: Tensor = self.embedding(tokens_batch_concated, offsets)

        # Shape [batch_size][class_count]
        class_logits: Tensor = self.fc(embeddings)

        return class_logits

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_index: int) -> Tensor:
        tokens_batch_concated, offset_batch, label_batch = batch

        criterion = CrossEntropyLoss().cuda()

        output_batch = self(tokens_batch_concated, offset_batch)
        loss = criterion(output_batch, label_batch)

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_index: int) -> None:
        tokens_batch_concated, offset_batch, label_batch = batch

        output_batch = self(tokens_batch_concated, offset_batch)

        # Update metric
        self.acc(output_batch, label_batch)

    def validation_epoch_end(self, outs) -> None:
        print('Accuracy', self.acc.compute())

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_index: int) -> None:
        self.validation_step(batch, batch_index)

    def test_epoch_end(self, outs) -> None:
        self.validation_epoch_end(outs)
