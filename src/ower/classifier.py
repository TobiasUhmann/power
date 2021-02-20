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
from torch.nn import EmbeddingBag, Linear, BCEWithLogitsLoss, Softmax, Parameter

from ower.util import log_tensor


class Classifier(LightningModule):
    embedding_bag: nn.EmbeddingBag
    linear: nn.Linear

    criterion: Any

    prec: pl.metrics.Precision
    recall: pl.metrics.Recall
    f1: pl.metrics.F1

    def __init__(self, vocab_size: int, emb_size: int, class_count: int):
        super().__init__()

        # Create layers
        self.embedding_bag = EmbeddingBag(vocab_size, emb_size)
        self.linear = Linear(class_count * emb_size, class_count)
        self.class_embs = Parameter(torch.rand((class_count, emb_size)))

        # Loss function
        self.criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([10] * 4))

        # Init weights
        initrange = 0.5
        self.embedding_bag.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

        # Add metrics
        self.prec = pl.metrics.Precision(num_classes=4, multilabel=True)
        self.recall = pl.metrics.Recall(num_classes=4, multilabel=True)
        self.f1 = pl.metrics.F1(num_classes=4, multilabel=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, sents_batch: Tensor) -> Tensor:
        """
        :param sents_batch: (batch_size, sent_count, sent_len)
        :return (batch_size, class_count)
        """

        log_path = f'{__name__}.{Classifier.__name__}.{Classifier.forward.__name__}'

        log_tensor(log_path, 'sents_batch', sents_batch,
                   '(batch_size = {}, sent_count = {}, sent_len = {})'.format(*sents_batch.shape))

        #
        # Embed sentences
        #
        # < sents_batch         (batch_size, sent_count, sent_len)
        # < embedding_bag       EmbeddingBag
        # > sent_embs_batch     (batch_size, sent_count, emb_size)
        #

        embedding_bag = self.embedding_bag

        log_tensor(log_path, 'embedding_bag.weight.data', embedding_bag.weight.data,
                   '(vocab_size = {}, emb_size = {})'.format(*embedding_bag.weight.data.shape))

        sent_embs_batch = self._embed_sents_batch(sents_batch, embedding_bag)

        log_tensor(log_path, 'sent_embs_batch', sent_embs_batch,
                   '(batch_size = {}, sent_count = {}, emb_size = {})'.format(*sent_embs_batch.shape))

        #
        # Calc attentions
        #
        # < sent_embs_batch     (batch_size, sent_count, emb_size)
        # < class_embs          (class_count, emb_size)
        # > softs_batch         (batch_size, class_count, sent_count)
        #

        class_embs = self.class_embs

        log_tensor(log_path, 'class_embs', class_embs,
                   '(class_count = {}, emb_size = {})'.format(*class_embs.shape))

        softs_batch = self._calc_attentions(sent_embs_batch, class_embs)

        log_tensor(log_path, 'softs_batch', softs_batch,
                   '(batch_size = {}, class_count = {}, sent_count = {})'.format(*softs_batch.shape))

        #
        # Weight sentences
        #
        # < sent_embs_batch     (batch_size, sent_count, emb_size)
        # < softs_batch         (batch_size, class_count, sent_count)
        # > weighted_batch      (batch_size, class_count, emb_size)
        #

        weighted_batch = self._weight_sents(sent_embs_batch, softs_batch)

        log_tensor(log_path, 'weighted_batch', weighted_batch,
                   '(batch_size = {}, class_count = {}, emb_size = {})'.format(*weighted_batch.shape))

        #
        # Concatenate weighted sentences
        #
        # < weighted_batch  (batch_size, class_count, emb_size)
        # > inputs_batch    (batch_size, class_count * emb_size)
        #

        batch_size, class_count, emb_size = weighted_batch.shape

        inputs_batch = weighted_batch.reshape(batch_size, class_count * emb_size)

        log_tensor(log_path, 'inputs_batch', inputs_batch,
                   '(batch_size = {}, class_count * emb_size = {})'.format(*inputs_batch.shape))

        #
        # Linear layer
        #
        # < inputs_batch    (batch_size, class_count * emb_size)
        # > outputs_batch   (batch_size, class_count)
        #

        linear = self.linear

        log_tensor(log_path, 'linear.weight.data', linear.weight.data,
                   '(input_dim = {}, output_dim = {})'.format(*linear.weight.data.shape))

        log_tensor(log_path, 'linear.bias.data', linear.bias.data,
                   '(output_dim = {})'.format(*linear.bias.data.shape))

        outputs_batch = self.linear(inputs_batch)

        log_tensor(log_path, 'outputs_batch', outputs_batch,
                   '(batch_size = {}, class_count = {})'.format(*outputs_batch.shape))

        return outputs_batch

    @staticmethod
    def _embed_sents_batch(sents_batch: Tensor, embedding_bag: EmbeddingBag) -> Tensor:
        """
        :param sents_batch: (batch_size, sent_count, sent_len)
        :return: (batch_size, sent_count, emb_size)
        """

        #
        # Flatten batch
        #
        # < sents_batch     (batch_size, sent_count, sent_len)
        # > flat_sents      (batch_size * sent_count, sent_len)
        #

        batch_size, sent_count, sent_len = sents_batch.shape

        flat_sents = sents_batch.reshape(batch_size * sent_count, sent_len)

        #
        # Embed sentences
        #
        # < flat_sents      (batch_size * sent_count, sent_len)
        # > flat_sent_embs  (batch_size * sent_count, emb_size)
        #

        flat_sent_embs = embedding_bag(flat_sents)

        #
        # Restore batch
        #
        # < flat_sent_embs      (batch_size * sent_count, emb_size)
        # > sent_embs_batch     (batch_size, sent_count, emb_size)
        #

        emb_size = flat_sent_embs.shape[-1]

        sent_embs_batch = flat_sent_embs.reshape(batch_size, sent_count, emb_size)

        return sent_embs_batch

    @staticmethod
    def _calc_attentions(sent_embs_batch: Tensor, class_embs: Tensor) -> Tensor:
        """
        :param sent_embs_batch: (batch_size, sent_count, emb_size)
        :return: (batch_size, class_count, sent_count)
        """

        #
        # Expand class embeddings for bmm()
        #
        # < class_embs          (class_count, emb_size)
        # > class_embs_batch    (batch_size, class_count, emb_size)
        #

        batch_size, sent_count, emb_size = sent_embs_batch.shape
        class_count, _emb_size = class_embs.shape

        class_embs_batch = class_embs.expand(batch_size, class_count, emb_size)

        #
        # Multiply each class with each sentence
        #
        # < class_embs_batch    (batch_size, class_count, emb_size)
        # < sent_embs_batch     (batch_size, sent_count, emb_size)
        # > atts_batch          (batch_size, class_count, sent_count)
        #

        atts_batch = torch.bmm(class_embs_batch, sent_embs_batch.transpose(1, 2))

        #
        # Apply softmax over sentences
        #
        # < atts_batch      (batch_size, class_count, sent_count)
        # > softs_batch     (batch_size, class_count, sent_count)
        #

        softs_batch = Softmax(dim=-1)(atts_batch)

        return softs_batch

    @staticmethod
    def _weight_sents(sent_embs_batch: Tensor, softs_batch: Tensor) -> Tensor:
        """
        :param sent_embs_batch: (batch_size, sent_count, emb_size)
        :param softs_batch: (batch_size, class_count, sent_count)
        :return: (batch_size, class_count, emb_size)
        """

        #
        # Repeat each batch slice class_count times
        #
        # < sent_embs_batch     (batch_size, sent_count, emb_size)
        # > expaned_batch       (batch_size, class_count, sent_count, emb_size)
        #

        batch_size, sent_count, emb_size = sent_embs_batch.shape
        _batch_size, class_count, _sent_count = softs_batch.shape

        expaned_batch = sent_embs_batch.unsqueeze(1).expand(-1, class_count, -1, -1)

        #
        # Flatten sentences for bmm()
        #
        # < expaned_batch   (batch_size, class_count, sent_count, emb_size)
        # > flat_expanded   (batch_size * class_count, sent_count, emb_size)
        #

        flat_expanded = expaned_batch.reshape(-1, sent_count, emb_size)

        #
        # Flatten attentions for bmm()
        #
        # < softs_batch     (batch_size, class_count, sent_count)
        # > flat_softs      (batch_size * class_count, sent_count, 1)
        #

        flat_softs = softs_batch.reshape(batch_size * class_count, sent_count).unsqueeze(-1)

        #
        # Multiply each sentence with each attention
        #
        # < flat_expanded   (batch_size * class_count, sent_count, emb_size)
        # < flat_softs      (batch_size * class_count, sent_count, 1)
        # > flat_weighted   (batch_size * class_count, emb_size)
        #

        flat_weighted = torch.bmm(flat_expanded.transpose(1, 2), flat_softs)

        #
        # Restore batch
        #
        # < flat_weighted   (batch_size * class_count, emb_size)
        # > weighted_batch  (batch_size, class_count, emb_size)
        #

        weighted_batch = flat_weighted.reshape(batch_size, class_count, emb_size)

        return weighted_batch

    def training_step(self, batch: Tuple[Tensor, Tensor], _batch_idx: int) -> Tensor:
        sents_batch, classes_batch = batch

        outputs_batch = self(sents_batch)
        loss = self.criterion(outputs_batch, classes_batch.float())

        self.log('loss (train)', loss.item())

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], _batch_index: int) -> None:
        sents_batch, classes_batch = batch

        outputs_batch = self(sents_batch)
        loss = self.criterion(outputs_batch, classes_batch.float())

        self.log('loss (valid)', loss.item())

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
