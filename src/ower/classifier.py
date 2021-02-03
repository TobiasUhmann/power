from typing import Tuple, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn import EmbeddingBag, Linear, BCEWithLogitsLoss, Softmax


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
        self.fc = Linear(num_classes * embed_dim, num_classes)

        self.class_embs = torch.rand((4, embed_dim), requires_grad=True).cuda()

        # Loss function
        self.criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([1] * 4))

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

    def forward(self, sents_batch: Tensor) -> Tensor:
        """
        :param sents_batch: shape = (batch_size, sents, sent_len)
        """

        batch_size = len(sents_batch)

        # shape = (batch_size * sents, sent_len)
        flat_sents = sents_batch.reshape(-1, 64)

        # shape = (batch_size * sents, emb_size)
        flat_sent_embs = self.embedding(flat_sents)

        # shape = (batch_size, sents, emb_size)
        sent_embs_batch = flat_sent_embs.reshape(batch_size, 3, 32)

        # shape = (batch_size, classes, emb_size)
        class_embs_batch = self.class_embs.expand(batch_size, -1, -1)

        # shape = (batch_size, classes, sents)
        atts_batch = torch.bmm(class_embs_batch, sent_embs_batch.transpose(1, 2))

        # shape = (batch_size, classes, sents)
        softs_batch = Softmax(dim=-1)(atts_batch)

        # shape = (batch_size * classes, sents, 1)
        flat_softs = softs_batch.reshape(batch_size * 4, -1).unsqueeze(-1)

        # shape = (batch_size, 1, sents, emb_size)
        stacked_sents = sent_embs_batch.unsqueeze(1)

        # shape = (batch_size, classes, sents, emb_size)
        stacked_sents_2 = stacked_sents.expand(-1, 4, -1, -1)

        # shape = (batch_size * classes, sents, emb_size)
        stacked_sents_3 = stacked_sents_2.reshape(batch_size * 4, 3, 32)

        # shape = (batch_size * classes, emb_size)
        flat_weighted = torch.bmm(stacked_sents_3.transpose(1, 2), flat_softs)

        # shape = (batch_size, classes, emb_size)
        weighted_batch = flat_weighted.reshape(batch_size, 4, -1)

        # shape = (batch_size, classes * emb_size)
        inputs_batch = weighted_batch.reshape(batch_size, -1)

        # shape = (batch_size, classes)
        outputs_batch = self.fc(inputs_batch)

        return outputs_batch

        # outputs = torch.empty((len(embeddings_1), len(self.class_embs), 32)).cuda()
        #
        # for k, (sent_1, sent_2, sent_3) in enumerate(zip(embeddings_1, embeddings_2, embeddings_3)):
        #
        #     sents = [sent_1, sent_2, sent_3]
        #     outer_prod = torch.empty((len(self.class_embs), len(sents))).cuda()
        #
        #     for i, class_embedding in enumerate(self.class_embs):
        #         for j, sent in enumerate(sents):
        #             outer_prod[i, j] = torch.dot(class_embedding, sent)
        #
        #     att_softs_batch = Softmax(dim=1)(outer_prod)
        #
        #     weighted_sents = torch.empty((len(self.class_embs), 32)).cuda()
        #
        #     for i, soft in enumerate(att_softs_batch):
        #         dot_prod = torch.zeros((32,)).cuda()
        #         for s, sent in zip(soft, sents):
        #             dot_prod += s.item() * sent
        #         weighted_sents[i] = dot_prod
        #
        #     outputs[k] = weighted_sents

    def training_step(self, batch: Tuple[Tensor, Tensor], _batch_idx: int) -> Tensor:
        sents_batch, classes_batch = batch

        outputs_batch = self(sents_batch)
        loss = self.criterion(outputs_batch, classes_batch.float())

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], _batch_index: int) -> None:
        sents_batch, classes_batch = batch

        outputs_batch = self(sents_batch)

        # Update metrics
        self.prec(outputs_batch, classes_batch)
        self.recall(outputs_batch, classes_batch)
        self.f1(outputs_batch, classes_batch)

    def validation_epoch_end(self, outs) -> None:
        print()
        print('Precision', self.prec.compute())
        print('Recall', self.recall.compute())
        print('F1', self.f1.compute())

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_index: int) -> None:
        self.validation_step(batch, batch_index)

    def test_epoch_end(self, outs) -> None:
        self.validation_epoch_end(outs)
