"""
The `OWER Classifier` takes an entity's sentences and maps them to the output classes.
It implements a simple attention mechanism to focus on sentences that relate to the
class in question.
"""

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

    def __init__(self, vocab_size: int, emb_size: int, num_classes: int):
        super().__init__()

        # Create layers
        self.embedding = EmbeddingBag(vocab_size, emb_size, sparse=True)
        self.fc = Linear(num_classes * emb_size, num_classes)

        self.class_embs = torch.rand((4, emb_size), requires_grad=True).cuda()

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
        :param sents_batch: Batch of entities, multiple sentences per entities, each sentence
                            as a token list of fixed length (padded)

                            shape = (batch_size, sent_count, sent_len)

        :return shape (batch_size, class_count)
        """

        batch_size, sent_count, sent_len = sents_batch.shape

        #
        # Embed sentences
        #

        # (batch_size, sent_count, sent_len)
        # -> (batch_size * sent_count, sent_len)
        flat_sents = sents_batch.reshape(batch_size * sent_count, sent_len)

        # (batch_size * sent_count, sent_len)
        # -> (batch_size * sent_count, emb_size)
        flat_sent_embs = self.embedding(flat_sents)
        emb_size = flat_sent_embs.shape[-1]

        # (batch_size * sent_count, emb_size)
        # -> (batch_size, sent_count, emb_size)
        sent_embs_batch = flat_sent_embs.reshape(batch_size, sent_count, emb_size)

        #
        # Calc attentions
        #

        # (class_count, emb_size)
        # -> (batch_size, class_count, emb_size)
        class_count = self.class_embs.shape[0]
        class_embs_batch = self.class_embs.expand(batch_size, class_count, emb_size)

        # (batch_size, class_count, emb_size) @ (batch_size, emb_size, sent_count)
        # -> (batch_size, class_count, sent_count)
        atts_batch = torch.bmm(class_embs_batch, sent_embs_batch.transpose(1, 2))

        # (batch_size, class_count, sent_count)
        # -> (batch_size, class_count, sent_count)
        softs_batch = Softmax(dim=-1)(atts_batch)

        #
        # Weight sentences
        #

        # (batch_size, sent_count, emb_size)
        # -> (batch_size, class_count, sent_count, emb_size)
        expaned_batch = sent_embs_batch.unsqueeze(1).expand(-1, class_count, -1, -1)

        # (batch_size, class_count, sent_count, emb_size)
        # -> (batch_size * class_count, sent_count, emb_size)
        flat_expanded = expaned_batch.reshape(-1, sent_count, emb_size)

        # (batch_size, class_count, sent_count)
        # -> (batch_size * class_count, sent_count)
        flat_softs = softs_batch.reshape(batch_size * class_count, sent_count)

        # (batch_size * class_count, emb_size, sent_count) @ (batch_size * class_count, sent_count, 1)
        # shape = (batch_size * class_count, emb_size)
        flat_weighted = torch.bmm(flat_expanded.transpose(1, 2), flat_softs.unsqueeze(-1))

        # (batch_size * class_count, emb_size)
        # -> (batch_size, class_count, emb_size)
        weighted_batch = flat_weighted.reshape(batch_size, class_count, emb_size)

        #
        # Classify weighted sentences
        #

        # (batch_size, class_count, emb_size)
        # -> (batch_size, class_count * emb_size)
        inputs_batch = weighted_batch.reshape(batch_size, class_count * emb_size)

        # (batch_size, class_count * emb_size)
        # -> (batch_size, class_count)
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
