import torch
from torch import Tensor
from torch.nn import Softmax, EmbeddingBag, Linear, Module, Parameter

from notebooks._04_classifier.util import log_tensor, get_word_lbls, get_emb_lbls, get_tok_lbls, get_sent_lbls, \
    get_ent_lbls, get_class_lbls, get_ent_class_lbls, get_mix_emb_lbls, get_ent_sent_lbls


class Classifier(Module):
    embedding_bag: EmbeddingBag
    linear: Linear
    class_embs: Parameter

    def __init__(self, vocab_size: int, emb_size: int, class_count: int):
        super().__init__()

        self.embedding_bag = EmbeddingBag(vocab_size, emb_size)
        self.linear = Linear(class_count * emb_size, class_count)
        self.class_embs = Parameter(torch.randn((class_count, emb_size)))

        # log_tensor(self.embedding_bag.weight, 'self.embedding_bag.weight', [get_word_lbls(), get_emb_lbls()])
        # log_tensor(self.class_embs, 'self.class_embs', [get_class_lbls(), get_emb_lbls()])
        # log_tensor(self.linear.weight.data.detach(), 'self.linear.weight.data', [get_class_lbls(), get_mix_emb_lbls()])
        # log_tensor(self.linear.bias.data.detach(), 'self.linear.bias.data', [get_class_lbls()])

        # Init weights
        initrange = 0.5
        self.embedding_bag.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.uniform_(-initrange, initrange)

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

        # log_tensor(self.embedding_bag.weight, 'self.embedding_bag.weight', [get_word_lbls(), get_emb_lbls()])
        # log_tensor(sents_batch, 'sents_batch', [get_ent_lbls(), get_sent_lbls(), get_tok_lbls()])
        # log_tensor(sent_embs_batch, 'sent_embs_batch', [get_ent_lbls(), get_sent_lbls(), get_emb_lbls()])

        #
        # Calculate attentions (which class matches which sentences)
        #
        # < class_embs       (class_count, emb_size)
        # < sent_embs_batch  (batch_size, sent_count, emb_size)
        # > atts_batch       (batch_size, class_count, sent_count)
        #

        atts_batch = self.calc_atts(sent_embs_batch)

        # log_tensor(self.class_embs, 'self.class_embs', [get_class_lbls(), get_emb_lbls()])
        # log_tensor(sent_embs_batch, 'sent_embs_batch', [get_ent_lbls(), get_sent_lbls(), get_emb_lbls()])
        # log_tensor(atts_batch, 'atts_batch', [get_ent_lbls(), get_class_lbls(), get_sent_lbls()])

        #
        # For each class, mix sentences (as per class' attentions to sentences)
        #
        # < sent_embs_batch  (batch_size, sent_count, emb_size)
        # < atts_batch       (batch_size, class_count, sent_count)
        # > mixes_batch      (batch_size, class_count, emb_size)
        #

        mixes_batch = self.mix_sents(sent_embs_batch, atts_batch)

        # log_tensor(sent_embs_batch, 'sent_embs_batch', [get_ent_lbls(), get_sent_lbls(), get_emb_lbls()])
        # log_tensor(atts_batch, 'atts_batch', [get_ent_lbls(), get_class_lbls(), get_sent_lbls()])
        # log_tensor(mixes_batch, 'mixes_batch', [get_ent_lbls(), get_class_lbls(), get_emb_lbls()])

        #
        # Concatenate mixes
        #
        # < mixes_batch         (batch_size, class_count, emb_size)
        # > concat_mixes_batch  (batch_size, class_count * emb_size)
        #

        batch_size, class_count, emb_size = mixes_batch.shape

        concat_mixes_batch = mixes_batch.reshape(batch_size, class_count * emb_size)

        # log_tensor(mixes_batch, 'mixes_batch', [get_ent_lbls(), get_class_lbls(), get_emb_lbls()])
        # log_tensor(concat_mixes_batch, 'concat_mixes_batch', [get_ent_lbls(), get_mix_emb_lbls()])

        #
        # Push concatenated mixes through linear layer
        #
        # < concat_mixes_batch  (batch_size, class_count * emb_size)
        # > logits_batch        (batch_size, class_count)
        #

        logits_batch = self.linear(concat_mixes_batch)

        # log_tensor(concat_mixes_batch, 'concat_mixes_batch', [get_ent_lbls(), get_mix_emb_lbls()])
        # log_tensor(logits_batch, 'logits_batch', [get_ent_lbls(), get_class_lbls()])

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

        # log_tensor(sents_batch, 'sents_batch', [get_ent_lbls(), get_sent_lbls(), get_tok_lbls()])
        # log_tensor(flat_sents, 'flat_sents', [get_ent_sent_lbls(), get_tok_lbls()])

        #
        # Embed sentences
        #
        # < embedding_bag.weight  (vocab_size, emb_size)
        # < flat_sents            (batch_size * sent_count, sent_len)
        # > flat_sent_embs        (batch_size * sent_count, emb_size)
        #

        flat_sent_embs = self.embedding_bag(flat_sents)

        # log_tensor(self.embedding_bag.weight, 'self.embedding_bag.weight', [get_word_lbls(), get_emb_lbls()])
        # log_tensor(flat_sents, 'flat_sents', [get_ent_sent_lbls(), get_tok_lbls()])
        # log_tensor(flat_sent_embs, 'flat_sent_embs', [get_ent_sent_lbls(), get_emb_lbls()])

        #
        # Restore batch
        #
        # < flat_sent_embs   (batch_size * sent_count, emb_size)
        # > sent_embs_batch  (batch_size, sent_count, emb_size)
        #

        _, emb_size = flat_sent_embs.shape

        sent_embs_batch = flat_sent_embs.reshape(batch_size, sent_count, emb_size)

        # log_tensor(flat_sent_embs, 'flat_sent_embs', [get_ent_sent_lbls(), get_emb_lbls()])
        # log_tensor(sent_embs_batch, 'sent_embs_batch', [get_ent_lbls(), get_sent_lbls(), get_emb_lbls()])

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

        # log_tensor(self.class_embs, 'self.class_embs', [get_class_lbls(), get_emb_lbls()])
        # log_tensor(class_embs_batch, 'class_embs_batch', [get_ent_lbls(), get_class_lbls(), get_emb_lbls()])

        #
        # Multiply each class with each sentence
        #
        # < class_embs_batch    (batch_size, class_count, emb_size)
        # < sent_embs_batch     (batch_size, sent_count, emb_size)
        # > atts_batch          (batch_size, class_count, sent_count)
        #

        atts_batch = torch.bmm(class_embs_batch, sent_embs_batch.transpose(1, 2))

        # log_tensor(class_embs_batch, 'class_embs_batch', [get_ent_lbls(), get_class_lbls(), get_emb_lbls()])
        # log_tensor(sent_embs_batch, 'sent_embs_batch', [get_ent_lbls(), get_sent_lbls(), get_emb_lbls()])
        # log_tensor(atts_batch, 'atts_batch', [get_ent_lbls(), get_class_lbls(), get_sent_lbls()])

        #
        # Apply softmax over sentences
        #
        # < atts_batch      (batch_size, class_count, sent_count)
        # > softs_batch     (batch_size, class_count, sent_count)
        #

        softs_batch = Softmax(dim=-1)(atts_batch)

        # log_tensor(atts_batch, 'atts_batch', [get_ent_lbls(), get_class_lbls(), get_sent_lbls()])
        # log_tensor(softs_batch, 'softs_batch', [get_ent_lbls(), get_class_lbls(), get_sent_lbls()])

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

        # log_tensor(sent_embs_batch, 'sent_embs_batch', [get_ent_lbls(), get_sent_lbls(), get_emb_lbls()])
        # log_tensor(expaned_batch, 'expaned_batch', [get_ent_lbls(), get_class_lbls(), get_sent_lbls(), get_emb_lbls()])

        #
        # Flatten sentences for bmm()
        #
        # < expaned_batch   (batch_size, class_count, sent_count, emb_size)
        # > flat_expanded   (batch_size * class_count, sent_count, emb_size)
        #

        _, _, sent_count, emb_size = expaned_batch.shape

        flat_expanded = expaned_batch.reshape(-1, sent_count, emb_size)

        # log_tensor(expaned_batch, 'expaned_batch', [get_ent_lbls(), get_class_lbls(), get_sent_lbls(), get_emb_lbls()])
        # log_tensor(flat_expanded, 'flat_expanded', [get_ent_class_lbls(), get_sent_lbls(), get_emb_lbls()])

        #
        # Flatten attentions for bmm()
        #
        # < atts_batch  (batch_size, class_count, sent_count)
        # > flat_atts   (batch_size * class_count, sent_count, 1)
        #

        batch_size, _, _ = atts_batch.shape

        flat_atts = atts_batch.reshape(batch_size * class_count, sent_count).unsqueeze(-1)

        # log_tensor(atts_batch, 'softs_batch', [get_ent_lbls(), get_class_lbls(), get_sent_lbls()])
        # log_tensor(flat_atts, 'flat_atts', [get_ent_class_lbls(), get_sent_lbls(), ['']])

        #
        # Multiply each sentence with each attention
        #
        # < flat_expanded  (batch_size * class_count, sent_count, emb_size)
        # < flat_atts      (batch_size * class_count, sent_count, 1)
        # > flat_mixes     (batch_size * class_count, emb_size)
        #

        flat_mixes = torch.bmm(flat_expanded.transpose(1, 2), flat_atts).squeeze(-1)

        # log_tensor(flat_expanded, 'flat_expanded', [get_ent_class_lbls(), get_sent_lbls(), get_emb_lbls()])
        # log_tensor(flat_atts, 'flat_atts', [get_ent_class_lbls(), get_sent_lbls(), ['']])
        # log_tensor(flat_mixes, 'flat_mixes', [get_ent_class_lbls(), get_emb_lbls()])

        #
        # Restore batch
        #
        # < flat_mixes   (batch_size * class_count, emb_size)
        # > mixes_batch  (batch_size, class_count, emb_size)
        #

        mixes_batch = flat_mixes.reshape(batch_size, class_count, emb_size)

        # log_tensor(flat_mixes, 'flat_mixes', [get_ent_class_lbls(), get_emb_lbls()])
        # log_tensor(mixes_batch, 'mixes_batch', [get_ent_lbls(), get_class_lbls(), get_emb_lbls()])

        return mixes_batch
