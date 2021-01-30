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

        self.class_embeddings = torch.rand((4, embed_dim), requires_grad=True).cuda()

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

    def forward(self,
                tokens_batch_concated_1: Tensor, offsets_batch_1: Tensor,
                tokens_batch_concated_2: Tensor, offsets_batch_2: Tensor,
                tokens_batch_concated_3: Tensor, offsets_batch_3: Tensor,
                ) -> Tensor:
        # Shape [batch_size][embed_dim]
        embeddings_1: Tensor = self.embedding(tokens_batch_concated_1, offsets_batch_1)
        embeddings_2: Tensor = self.embedding(tokens_batch_concated_2, offsets_batch_2)
        embeddings_3: Tensor = self.embedding(tokens_batch_concated_3, offsets_batch_3)

        logits_batch = torch.empty((len(embeddings_1), len(self.class_embeddings), 32)).cuda()

        for k, (sent_1, sent_2, sent_3) in enumerate(zip(embeddings_1, embeddings_2, embeddings_3)):

            sents = [sent_1, sent_2, sent_3]
            outer_prod = torch.empty((len(self.class_embeddings), len(sents))).cuda()

            for i, class_embedding in enumerate(self.class_embeddings):
                for j, sent in enumerate(sents):
                    outer_prod[i, j] = torch.dot(class_embedding, sent)

            softs = Softmax(dim=1)(outer_prod)

            weighted_sents = torch.empty((len(self.class_embeddings),32)).cuda()

            for i, soft in enumerate(softs):
                dot_prod = torch.zeros((32,)).cuda()
                for s, sent in zip(soft, sents):
                    dot_prod += s.item() * sent
                weighted_sents[i] = dot_prod

            logits_batch[k] = weighted_sents

        # Shape [batch_size][class_count]
        outputs_batch: Tensor = self.fc(torch.flatten(logits_batch, start_dim=1, end_dim=-1))

        return outputs_batch

    def training_step(self,
                      batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
                      batch_idx: int
                      ) -> Tensor:
        tokens_batch_concated_1, offset_batch_1, \
        tokens_batch_concated_2, offset_batch_2, \
        tokens_batch_concated_3, offset_batch_3, \
        classes_batch = batch

        class_logits_batch = self(
            tokens_batch_concated_1, offset_batch_1,
            tokens_batch_concated_2, offset_batch_2,
            tokens_batch_concated_3, offset_batch_3,
        )

        loss = self.criterion(class_logits_batch, classes_batch.float())

        return loss

    def validation_step(self,
                        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
                        batch_idx: int
                        ) -> None:
        tokens_batch_concated_1, offset_batch_1, \
        tokens_batch_concated_2, offset_batch_2, \
        tokens_batch_concated_3, offset_batch_3, \
        classes_batch = batch

        class_logits_batch = self(
            tokens_batch_concated_1, offset_batch_1,
            tokens_batch_concated_2, offset_batch_2,
            tokens_batch_concated_3, offset_batch_3,
        )

        # Update metrics
        self.prec(class_logits_batch, classes_batch)
        self.recall(class_logits_batch, classes_batch)
        self.f1(class_logits_batch, classes_batch)

    def validation_epoch_end(self, outs) -> None:
        print()
        print('Precision', self.prec.compute())
        print('Recall', self.recall.compute())
        print('F1', self.f1.compute())

    def test_step(self,
                  batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
                  batch_index: int
                  ) -> None:
        self.validation_step(batch, batch_index)

    def test_epoch_end(self, outs) -> None:
        self.validation_epoch_end(outs)
