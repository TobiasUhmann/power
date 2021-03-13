import torch
from torch import Tensor
from torch.nn import Module, EmbeddingBag, Parameter, Softmax
from torchtext.vocab import Vocab


class Classifier(Module):

    embedding_bag: EmbeddingBag
    class_embs: Parameter
    multi_weight: Parameter
    multi_bias: Parameter

    def __init__(self, embedding_bag: EmbeddingBag, class_count: int):
        super().__init__()

        self.embedding_bag = embedding_bag

        _, emb_size = embedding_bag.weight.data.shape
        self.class_embs = Parameter(torch.randn(class_count, emb_size))
        self.multi_weight = Parameter(torch.randn(class_count, emb_size))
        self.multi_bias = Parameter(torch.randn(class_count, 1))

    @classmethod
    def from_random(cls, vocab_size: int, emb_size: int, class_count: int):
        embedding_bag = EmbeddingBag(num_embeddings=vocab_size, embedding_dim=emb_size)

        return cls(embedding_bag, class_count)

    @classmethod
    def from_pre_trained(cls, vocab: Vocab, class_count: int):
        embedding_bag = EmbeddingBag.from_pretrained(vocab.vectors)

        return cls(embedding_bag, class_count)

    def forward(self, tok_lists_batch: Tensor) -> Tensor:
        """
        :param tok_lists_batch: (batch_size, sent_count, sent_len)
        :return (batch_size, class_count)
        """

        #
        # Embed token lists
        #
        # < tok_lists_batch  (batch_size, sent_count, sent_len)
        # > sents_batch      (batch_size, sent_count, emb_size)
        #

        sents_batch = self._embed_tok_lists(tok_lists_batch)

        #
        # Calculate attentions (which class matches which sentences)
        #
        # < sents_batch  (batch_size, sent_count, emb_size)
        # > atts_batch   (batch_size, class_count, sent_count)
        #

        atts_batch = self._calc_atts(sents_batch)

        #
        # For each class, mix sentences according to attention
        #
        # < atts_batch   (batch_size, class_count, sent_count)
        # < sents_batch  (batch_size, sent_count, emb_size)
        # > mixes_batch  (batch_size, class_count, emb_size)
        #

        mixes_batch = torch.bmm(atts_batch, sents_batch)

        #
        # Push mixes through linear layers
        #
        # < concat_mixes_batch  (batch_size, class_count * emb_size)
        # > logits_batch        (batch_size, class_count)
        #

        logits_batch = self._multi_linear(mixes_batch)

        return logits_batch

    def _embed_tok_lists(self, tok_lists_batch: Tensor) -> Tensor:
        """
        :param tok_lists_batch: (batch_size, sent_count, sent_len)
        :return: (batch_size, sent_count, emb_size)
        """

        #
        # Flatten batch
        #
        # < tok_lists_batch  (batch_size, sent_count, sent_len)
        # > flat_tok_lists   (batch_size * sent_count, sent_len)
        #

        batch_size, sent_count, sent_len = tok_lists_batch.shape

        flat_tok_lists = tok_lists_batch.reshape(batch_size * sent_count, sent_len)

        #
        # Embed token lists
        #
        # < flat_tok_lists  (batch_size * sent_count, sent_len)
        # > flat_sents      (batch_size * sent_count, emb_size)
        #

        flat_sents = self.embedding_bag(flat_tok_lists)

        #
        # Restore batch
        #
        # < flat_sents   (batch_size * sent_count, emb_size)
        # > sents_batch  (batch_size, sent_count, emb_size)
        #

        _, emb_size = flat_sents.shape

        sents_batch = flat_sents.reshape(batch_size, sent_count, emb_size)

        return sents_batch

    def _calc_atts(self, sents_batch: Tensor) -> Tensor:
        """
        :param sents_batch: (batch_size, sent_count, emb_size)
        :return: (batch_size, class_count, sent_count)
        """

        #
        # Expand class embeddings for bmm()
        #
        # < self.class_embs   (class_count, emb_size)
        # > class_embs_batch  (batch_size, class_count, emb_size)
        #

        batch_size, _, _ = sents_batch.shape

        class_embs_batch = self.class_embs.unsqueeze(0).expand(batch_size, -1, -1)

        #
        # Multiply each class with each sentence
        #
        # < class_embs_batch  (batch_size, class_count, emb_size)
        # < sents_batch       (batch_size, sent_count, emb_size)
        # > atts_batch        (batch_size, class_count, sent_count)
        #

        atts_batch = torch.bmm(class_embs_batch, sents_batch.transpose(1, 2))

        #
        # Softmax over sentences
        #
        # < atts_batch   (batch_size, class_count, sent_count)
        # > softs_batch  (batch_size, class_count, sent_count)
        #

        softs_batch = Softmax(dim=-1)(atts_batch)

        return softs_batch

    def _multi_linear(self, mixes_batch: Tensor) -> Tensor:
        """
        Push each sentence mix through its respective linear layer.

        For example, push the "married" mix through the "married" layer
        that predicts the "married" class for the mix.

        :param mixes_batch: (batch_size, class_count, emb_size)
        :return: (batch_size, class_count)
        """

        #
        # Transpose per entity mixes -> per class mixes
        #
        # < mixes_batch  (batch_size, class_count, emb_size)
        # > mixes_batch  (class_count, batch_size, emb_size)
        #

        mixes_batch = mixes_batch.transpose(0, 1)

        #
        # Per class weight/bias row vector -> col vector
        #
        # < self.multi_weight  (class_count, emb_size)
        # < self.multi_bias    (class_count, 1)
        # > col_multi_weight   (class_count, emb_size, 1)
        # > col_multi_bias     (class_count, 1, 1)
        #

        col_multi_weight = self.multi_weight.unsqueeze(-1)
        col_multi_bias = self.multi_bias.unsqueeze(-1)

        #
        # Linear layers
        #
        # < mixes_batch       (class_count, batch_size, emb_size)
        # < col_multi_weight  (class_count, emb_size, 1)
        # < col_multi_bias    (class_count, 1, 1)
        # > logits_batch      (class_count, batch_size, 1)
        #

        logits_batch = torch.bmm(mixes_batch, col_multi_weight) + col_multi_bias

        #
        # Per class outputs col vector -> row vector & Restore per entity batch
        #
        # < logits_batch  (class_count, batch_size, 1)
        # > logits_batch  (batch_size, class_count)
        #

        logits_batch = logits_batch.squeeze(-1).T

        return logits_batch
