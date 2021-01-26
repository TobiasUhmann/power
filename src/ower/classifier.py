from typing import Tuple, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn import EmbeddingBag, Linear, BCEWithLogitsLoss


class Classifier(LightningModule):
    embedding: nn.EmbeddingBag
    fc: nn.Linear

    criterion: Any

    prec: pl.metrics.Precision
    recall: pl.metrics.Recall
    f1: pl.metrics.F1

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()

        # Create layers
        self.embedding = EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = Linear(embed_dim, num_classes)

        # Loss function
        self.criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([10] * 4))

        # Init weights
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

        # Add metrics
        self.prec = pl.metrics.Precision(num_classes=4, multilabel=True)
        self.recall = pl.metrics.Recall(num_classes=4, multilabel=True)
        self.f1 = pl.metrics.F1(num_classes=4, multilabel=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=4.0)

    def forward(self, tokens_batch_concated: Tensor, offsets_batch: Tensor):
        # Shape [batch_size][embed_dim]
        embeddings: Tensor = self.embedding(tokens_batch_concated, offsets_batch)

        # Shape [batch_size][class_count]
        class_logits: Tensor = self.fc(embeddings)

        return class_logits

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        tokens_batch_concated, offset_batch, classes_batch = batch

        class_logits_batch = self(tokens_batch_concated, offset_batch)
        loss = self.criterion(class_logits_batch, classes_batch.float())

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        tokens_batch_concated, offset_batch, classes_batch = batch

        class_logits_batch = self(tokens_batch_concated, offset_batch)

        # Update metrics
        self.prec(class_logits_batch, classes_batch)
        self.recall(class_logits_batch, classes_batch)
        self.f1(class_logits_batch, classes_batch)

    def validation_epoch_end(self, outs) -> None:
        print()
        print('Precision', self.prec.compute())
        print('Recall', self.recall.compute())
        print('F1', self.f1.compute())

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_index: int) -> None:
        self.validation_step(batch, batch_index)

    def test_epoch_end(self, outs) -> None:
        self.validation_epoch_end(outs)
