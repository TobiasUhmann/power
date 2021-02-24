import torch
from torch import Tensor
from torch.nn import Module, EmbeddingBag, Linear, Parameter, Softmax

from ower.util import log_tensor


class Classifier(Module):

    embedding_bag: EmbeddingBag
    linear: Linear
    class_embs: Parameter

    def __init__(self, vocab_size: int, emb_size: int, class_count: int):
        super().__init__()

        self.embedding_bag = EmbeddingBag(vocab_size, emb_size)
        self.linear = Linear(class_count * emb_size, class_count)
        self.class_embs = Parameter(torch.randn((class_count, emb_size)))

        # Init weights
        initrange = 0.5
        self.embedding_bag.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, sents_batch: Tensor) -> Tensor:
        """
        :param sents_batch: (batch_size, sent_count, sent_len)
        :return (batch_size, class_count)
        """

        #
        # Embed sentences
        #
        # < embedding_bag.weight  (vocab_size, emb_size)
        # < sents_batch           (batch_size, sent_count, sent_len)
        # > sent_embs_batch       (batch_size, sent_count, emb_size)
        #

        sent_embs_batch = self.embed_sents(sents_batch)

        #
        # Calculate attentions (which class matches which sentences)
        #
        # < class_embs       (class_count, emb_size)
        # < sent_embs_batch  (batch_size, sent_count, emb_size)
        # > atts_batch       (batch_size, class_count, sent_count)
        #

        atts_batch = self.calc_atts(sent_embs_batch)

        #
        # For each class, mix sentences (as per class' attentions to sentences)
        #
        # < sent_embs_batch  (batch_size, sent_count, emb_size)
        # < atts_batch       (batch_size, class_count, sent_count)
        # > mixes_batch      (batch_size, class_count, emb_size)
        #

        mixes_batch = self.mix_sents(atts_batch)

        #
        # Concatenate mixes
        #
        # < weighted_batch  (batch_size, class_count, emb_size)
        # > concat_mixes_batch  (batch_size, class_count * emb_size)
        #

        concat_mixes_batch = mixes_batch.reshape(batch_size, class_count * emb_size)

        #
        # Push concatenated mixes through linear layer
        #
        # < concat_mixes_batch  (batch_size, class_count * emb_size)
        # > logits_batch        (batch_size, class_count)
        #

        logits_batch = self.linear(concat_mixes_batch)

        return logits_batch

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
