import torch
from torch import Tensor
from torch.nn import EmbeddingBag, Module, Softmax, Parameter
from torchtext.vocab import Vocab


class Classifier(Module):
    embedding_bag: EmbeddingBag
    class_embs: Parameter
    multi_weight: Parameter
    multi_bias: Parameter

    def __init__(self,
                 embedding_bag: EmbeddingBag,
                 class_embs: Parameter,
                 multi_weight: Parameter,
                 multi_bias: Parameter):
        super().__init__()

        self.embedding_bag = embedding_bag
        self.class_embs = class_embs
        self.multi_weight = multi_weight
        self.multi_bias = multi_bias

    @classmethod
    def from_random(cls, vocab: Vocab, emb_size: int, class_count: int):
        embedding_bag = EmbeddingBag(num_embeddings=len(vocab), embedding_dim=emb_size)
        class_embs = Parameter(torch.randn(class_count, emb_size))
        multi_weight = Parameter(torch.randn(class_count, emb_size, 2))
        multi_bias = Parameter(torch.randn(class_count, 1, 2))

        return cls(embedding_bag, class_embs, multi_weight, multi_bias)

    @classmethod
    def from_pre_trained(cls, vocab: Vocab, class_count: int, freeze=True):
        embedding_bag = EmbeddingBag.from_pretrained(vocab.vectors, freeze=freeze)

        emb_size = vocab.vectors.shape[1]
        class_embs = Parameter(torch.randn(class_count, emb_size))
        multi_weight = Parameter(torch.randn(class_count, emb_size, 2))
        multi_bias = Parameter(torch.randn(class_count, 1, 2))

        return cls(embedding_bag, class_embs, multi_weight, multi_bias)

    def forward(self, sents_batch: Tensor) -> Tensor:
        """
        :param sents_batch: (batch_size, sent_count, sent_len)
        :return (batch_size, class_count, 2)
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

        mixes_batch = self.mix_sents(sent_embs_batch, atts_batch)

        #
        # Push mixes through respective linear layers
        #
        # < mixes_batch   (batch_size, class_count, emb_size)
        # > logits_batch  (batch_size, class_count, 2)
        #

        logits_batch = torch.bmm(mixes_batch.transpose(0, 1), self.multi_weight) + self.multi_bias
        logits_batch = logits_batch.transpose(0, 1)

        return logits_batch

    def embed_sents(self, sents_batch: Tensor) -> Tensor:
        """
        :param sents_batch: (batch_size, sent_count, sent_len)
        :return: (batch_size, sent_count, emb_size)
        """

        #
        # Flatten batch
        #
        # < sents_batch  (batch_size, sent_count, sent_len)
        # > flat_sents   (batch_size * sent_count, sent_len)
        #

        batch_size, sent_count, sent_len = sents_batch.shape

        flat_sents = sents_batch.reshape(batch_size * sent_count, sent_len)

        #
        # Embed sentences
        #
        # < embedding_bag.weight  (vocab_size, emb_size)
        # < flat_sents            (batch_size * sent_count, sent_len)
        # > flat_sent_embs        (batch_size * sent_count, emb_size)
        #

        flat_sent_embs = self.embedding_bag(flat_sents)

        #
        # Restore batch
        #
        # < flat_sent_embs   (batch_size * sent_count, emb_size)
        # > sent_embs_batch  (batch_size, sent_count, emb_size)
        #

        _, emb_size = flat_sent_embs.shape

        sent_embs_batch = flat_sent_embs.reshape(batch_size, sent_count, emb_size)

        return sent_embs_batch

    def calc_atts(self, sent_embs_batch: Tensor) -> Tensor:
        """
        :param sent_embs_batch: (batch_size, sent_count, emb_size)
        :return: (batch_size, class_count, sent_count)
        """

        #
        # Expand class embeddings for bmm()
        #
        # < class_embs        (class_count, emb_size)
        # > class_embs_batch  (batch_size, class_count, emb_size)
        #

        batch_size, _, emb_size = sent_embs_batch.shape
        class_count, _ = self.class_embs.shape

        class_embs_batch = self.class_embs.expand(batch_size, class_count, emb_size)

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
    def mix_sents(sent_embs_batch: Tensor, atts_batch: Tensor) -> Tensor:
        """
        :param sent_embs_batch: (batch_size, sent_count, emb_size)
        :param atts_batch: (batch_size, class_count, sent_count)
        :return: (batch_size, class_count, emb_size)
        """

        #
        # Repeat each batch slice class_count times
        #
        # < sent_embs_batch     (batch_size, sent_count, emb_size)
        # > expaned_batch       (batch_size, class_count, sent_count, emb_size)
        #

        _, class_count, _ = atts_batch.shape

        expaned_batch = sent_embs_batch.unsqueeze(1).expand(-1, class_count, -1, -1)

        #
        # Flatten sentences for bmm()
        #
        # < expaned_batch   (batch_size, class_count, sent_count, emb_size)
        # > flat_expanded   (batch_size * class_count, sent_count, emb_size)
        #

        _, _, sent_count, emb_size = expaned_batch.shape

        flat_expanded = expaned_batch.reshape(-1, sent_count, emb_size)

        #
        # Flatten attentions for bmm()
        #
        # < atts_batch  (batch_size, class_count, sent_count)
        # > flat_atts   (batch_size * class_count, sent_count, 1)
        #

        batch_size, _, _ = atts_batch.shape

        flat_atts = atts_batch.reshape(batch_size * class_count, sent_count).unsqueeze(-1)

        #
        # Multiply each sentence with each attention
        #
        # < flat_expanded  (batch_size * class_count, sent_count, emb_size)
        # < flat_atts      (batch_size * class_count, sent_count, 1)
        # > flat_mixes     (batch_size * class_count, emb_size)
        #

        flat_mixes = torch.bmm(flat_expanded.transpose(1, 2), flat_atts).squeeze(-1)

        #
        # Restore batch
        #
        # < flat_mixes   (batch_size * class_count, emb_size)
        # > mixes_batch  (batch_size, class_count, emb_size)
        #

        mixes_batch = flat_mixes.reshape(batch_size, class_count, emb_size)

        return mixes_batch
