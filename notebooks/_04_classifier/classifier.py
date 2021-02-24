from typing import List

import torch
from torch import Tensor
from torch.nn import Softmax, EmbeddingBag, Linear

from notebooks._04_classifier.util import log_tensor

batch_size: int
class_count: int
emb_size: int
sent_count: int
sent_len: int

class_labels: List[str]
emb_labels: List[str]
ent_class_labels: List[str]
ent_labels: List[str]
ent_sent_labels: List[str]
mix_emb_labels: List[str]
sent_labels: List[str]
tok_labels: List[str]
word_labels: List[str]


def embed_sents(embedding_bag: EmbeddingBag, sents_batch: Tensor) -> Tensor:
    #
    # Flatten batch
    #
    # < sents_batch  (batch_size, sent_count, sent_len)
    # > flat_sents   (batch_size * sent_count, sent_len)
    #

    flat_sents = sents_batch.reshape(batch_size * sent_count, sent_len)

    # log_tensor(sents_batch, 'sents_batch', [ent_labels, sent_labels, tok_labels])
    # log_tensor(flat_sents, 'flat_sents', [ent_sent_labels, tok_labels])

    #
    # Embed sentences
    #
    # < embedding_bag.weight  (vocab_size, emb_size)
    # < flat_sents            (batch_size * sent_count, sent_len)
    # > flat_sent_embs        (batch_size * sent_count, emb_size)
    #

    flat_sent_embs = embedding_bag(flat_sents)

    # log_tensor(embedding_bag.weight, 'embedding_bag.weight', [word_labels, emb_labels])
    # log_tensor(flat_sents, 'flat_sents', [ent_sent_labels, tok_labels])
    # log_tensor(flat_sent_embs, 'flat_sent_embs', [ent_sent_labels, emb_labels])

    #
    # Restore batch
    #
    # < flat_sent_embs   (batch_size * sent_count, emb_size)
    # > sent_embs_batch  (batch_size, sent_count, emb_size)
    #

    sent_embs_batch = flat_sent_embs.reshape(batch_size, sent_count, emb_size)

    # log_tensor(flat_sent_embs, 'flat_sent_embs', [ent_sent_labels, emb_labels])
    # log_tensor(sent_embs_batch, 'sent_embs_batch', [ent_labels, sent_labels, emb_labels])

    return sent_embs_batch


def calc_atts(class_embs: Tensor, sent_embs_batch: Tensor) -> Tensor:
    #
    # Expand class embeddings for bmm()
    #
    # < class_embs        (class_count, emb_size)
    # > class_embs_batch  (batch_size, class_count, emb_size)
    #

    class_embs_batch = class_embs.expand(batch_size, class_count, emb_size)

    # log_tensor(class_embs, 'class_embs', [class_labels, emb_labels])
    # log_tensor(class_embs_batch, 'class_embs_batch', [ent_labels, class_labels, emb_labels])

    #
    # Multiply each class with each sentence
    #
    # < class_embs_batch    (batch_size, class_count, emb_size)
    # < sent_embs_batch     (batch_size, sent_count, emb_size)
    # > atts_batch          (batch_size, class_count, sent_count)
    #

    atts_batch = torch.bmm(class_embs_batch, sent_embs_batch.transpose(1, 2))

    # log_tensor(class_embs_batch, 'class_embs_batch', [ent_labels, class_labels, emb_labels])
    # log_tensor(sent_embs_batch, 'sent_embs_batch', [ent_labels, sent_labels, emb_labels])
    # log_tensor(atts_batch, 'atts_batch', [ent_labels, class_labels, sent_labels])

    #
    # Apply softmax over sentences
    #
    # < atts_batch      (batch_size, class_count, sent_count)
    # > softs_batch     (batch_size, class_count, sent_count)
    #

    softs_batch = Softmax(dim=-1)(atts_batch)

    # log_tensor(atts_batch, 'atts_batch', [ent_labels, class_labels, sent_labels])
    # log_tensor(softs_batch, 'softs_batch', [ent_labels, class_labels, sent_labels])

    return softs_batch


def mix_sents(sent_embs_batch: Tensor, softs_batch: Tensor) -> Tensor:
    #
    # Repeat each batch slice class_count times
    #
    # < sent_embs_batch     (batch_size, sent_count, emb_size)
    # > expaned_batch       (batch_size, class_count, sent_count, emb_size)
    #

    expaned_batch = sent_embs_batch.unsqueeze(1).expand(-1, class_count, -1, -1)

    # log_tensor(sent_embs_batch, 'sent_embs_batch', [ent_labels, sent_labels, emb_labels])
    # log_tensor(expaned_batch, 'expaned_batch', [ent_labels, class_labels, sent_labels, emb_labels])

    #
    # Flatten sentences for bmm()
    #
    # < expaned_batch   (batch_size, class_count, sent_count, emb_size)
    # > flat_expanded   (batch_size * class_count, sent_count, emb_size)
    #

    flat_expanded = expaned_batch.reshape(-1, sent_count, emb_size)

    # log_tensor(expaned_batch, 'expaned_batch', [ent_labels, class_labels, sent_labels, emb_labels])
    # log_tensor(flat_expanded, 'flat_expanded', [ent_class_labels, sent_labels, emb_labels])

    #
    # Flatten attentions for bmm()
    #
    # < softs_batch     (batch_size, class_count, sent_count)
    # > flat_softs      (batch_size * class_count, sent_count, 1)
    #

    flat_softs = softs_batch.reshape(batch_size * class_count, sent_count).unsqueeze(-1)

    # log_tensor(softs_batch, 'softs_batch', [ent_labels, class_labels, sent_labels])
    # log_tensor(flat_softs, 'flat_softs', [ent_class_labels, sent_labels, ['']])

    #
    # Multiply each sentence with each attention
    #
    # < flat_expanded  (batch_size * class_count, sent_count, emb_size)
    # < flat_softs     (batch_size * class_count, sent_count, 1)
    # > flat_mixes     (batch_size * class_count, emb_size)
    #

    flat_mixes = torch.bmm(flat_expanded.transpose(1, 2), flat_softs).squeeze(-1)

    # log_tensor(flat_expanded, 'flat_expanded', [ent_class_labels, sent_labels, emb_labels])
    # log_tensor(flat_softs, 'flat_softs', [ent_class_labels, sent_labels, ['']])
    # log_tensor(flat_mixes, 'flat_mixes', [ent_class_labels, emb_labels])

    #
    # Restore batch
    #
    # < flat_mixes   (batch_size * class_count, emb_size)
    # > mixes_batch  (batch_size, class_count, emb_size)
    #

    mixes_batch = flat_mixes.reshape(batch_size, class_count, emb_size)

    # log_tensor(flat_mixes, 'flat_mixes', [ent_class_labels, emb_labels])
    # log_tensor(mixes_batch, 'mixes_batch', [ent_labels, class_labels, emb_labels])

    return mixes_batch


def forward(embedding_bag: EmbeddingBag, class_embs: Tensor, linear: Linear, sents_batch: Tensor) -> Tensor:
    #
    # Embed sentences
    #
    # < embedding_bag.weight  (vocab_size, emb_size)
    # < sents_batch           (batch_size, sent_count, sent_len)
    # > sent_embs_batch       (batch_size, sent_count, emb_size)
    #

    sent_embs_batch = embed_sents(embedding_bag, sents_batch)

    # log_tensor(embedding_bag.weight, 'embedding_bag.weight', [word_labels, emb_labels])
    # log_tensor(sents_batch, 'sents_batch', [ent_labels, sent_labels, tok_labels])
    # log_tensor(sent_embs_batch, 'sent_embs_batch', [ent_labels, sent_labels, emb_labels])

    #
    # Calculate attentions (which class matches which sentences)
    #
    # < class_embs       (class_count, emb_size)
    # < sent_embs_batch  (batch_size, sent_count, emb_size)
    # > atts_batch       (batch_size, class_count, sent_count)
    #

    atts_batch = calc_atts(class_embs, sent_embs_batch)

    # log_tensor(class_embs, 'class_embs', [class_labels, emb_labels])
    # log_tensor(sent_embs_batch, 'sent_embs_batch', [ent_labels, sent_labels, emb_labels])
    # log_tensor(atts_batch, 'atts_batch', [ent_labels, class_labels, sent_labels])

    #
    # For each class, mix sentences (as per class' attentions to sentences)
    #
    # < sent_embs_batch  (batch_size, sent_count, emb_size)
    # < atts_batch       (batch_size, class_count, sent_count)
    # > mixes_batch      (batch_size, class_count, emb_size)
    #

    mixes_batch = mix_sents(sent_embs_batch, atts_batch)

    # log_tensor(sent_embs_batch, 'sent_embs_batch', [ent_labels, sent_labels, emb_labels])
    # log_tensor(atts_batch, 'atts_batch', [ent_labels, class_labels, sent_labels])
    # log_tensor(mixes_batch, 'mixes_batch', [ent_labels, class_labels, emb_labels])

    #
    # Concatenate mixes
    #
    # < weighted_batch  (batch_size, class_count, emb_size)
    # > concat_mixes_batch  (batch_size, class_count * emb_size)
    #

    concat_mixes_batch = mixes_batch.reshape(batch_size, class_count * emb_size)

    # log_tensor(mixes_batch, 'mixes_batch', [ent_labels, class_labels, emb_labels])
    # log_tensor(concat_mixes_batch, 'concat_mixes_batch', [ent_labels, mix_emb_labels])

    #
    # Push concatenated mixes through linear layer
    #
    # < concat_mixes_batch  (batch_size, class_count * emb_size)
    # > logits_batch        (batch_size, class_count)
    #

    logits_batch = linear(concat_mixes_batch)

    # log_tensor(concat_mixes_batch, 'concat_mixes_batch', [ent_labels, mix_emb_labels])
    # log_tensor(logits_batch, 'logits_batch', [ent_labels, class_labels])

    return logits_batch
