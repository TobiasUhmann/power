import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import nn


class Classifier(LightningModule):

    embedding: nn.EmbeddingBag
    fc: nn.Linear

    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()

        # Create layers
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

        # Init weights
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

        # Add metrics
        self.prec = pl.metrics.Precision(num_classes=100, multilabel=True)
        self.recall = pl.metrics.Recall(num_classes=100, multilabel=True)
        self.f1 = pl.metrics.F1(num_classes=100, multilabel=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=4.0)

    def forward(self, text_batch, offsets_batch):
        embedded = self.embedding(text_batch, offsets_batch)

        return self.fc(embedded)

    def training_step(self, batch, batch_idx):
        text_batch, offsets_batch, labels_batch = batch

        criterion = nn.BCEWithLogitsLoss()

        output_batch = self(text_batch, offsets_batch)
        loss = criterion(output_batch, labels_batch)

        # Update metrics
        self.prec(output_batch, labels_batch)
        self.recall(output_batch, labels_batch)
        self.f1(output_batch, labels_batch)

        return loss

    def training_epoch_end(self, outs):
        print()
        print('Precision', self.prec.compute())
        print('Recall', self.recall.compute())
        print('F1', self.f1.compute())