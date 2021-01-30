import os
from typing import List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torchtext.data import TabularDataset, Field
from torchtext.vocab import Vocab


class DataModule(LightningDataModule):
    data_dir: str
    num_classes: int
    num_sentences: int
    batch_size: int

    vocab: Vocab

    train_dataset: List[Tuple[int, List[int], List[List[int]]]]
    valid_dataset: List[Tuple[int, List[int], List[List[int]]]]
    test_dataset: List[Tuple[int, List[int], List[List[int]]]]

    def __init__(self, data_dir: str, num_classes: int, num_sentences: int, batch_size: int):
        super().__init__()

        self.data_dir = data_dir
        self.num_classes = num_classes
        self.num_sentences = num_sentences
        self.batch_size = batch_size

    #
    # Prepare data
    #

    def prepare_data(self):
        train_samples_tsv = os.path.join(self.data_dir, 'samples-v2-train.tsv')
        valid_samples_tsv = os.path.join(self.data_dir, 'samples-v2-valid.tsv')
        test_samples_tsv = os.path.join(self.data_dir, 'samples-v2-test.tsv')

        def tokenize(text: str):
            return text.split()

        fields = [('entity', Field(sequential=False, use_vocab=False))] + \
                 [(f'class_{i}', Field(sequential=False, use_vocab=False))
                  for i in range(self.num_classes)] + \
                 [(f'sent_{i}', Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True))
                  for i in range(self.num_sentences)]

        raw_train_set = TabularDataset(train_samples_tsv, 'tsv', fields, skip_header=True)
        raw_valid_set = TabularDataset(valid_samples_tsv, 'tsv', fields, skip_header=True)
        raw_test_set = TabularDataset(test_samples_tsv, 'tsv', fields, skip_header=True)

        last_sentence_field = fields[-1][1]
        last_sentence_field.build_vocab(raw_train_set)
        self.vocab = last_sentence_field.vocab

        self.train_dataset = [(int(getattr(sample, 'entity')),
                               [int(getattr(sample, f'class_{i}'))
                                for i in range(self.num_classes)],
                               [[self.vocab[token] for token in getattr(sample, f'sent_{i}')]
                                for i in range(self.num_sentences)])
                              for sample in raw_train_set]

        self.valid_dataset = [(int(getattr(sample, 'entity')),
                               [int(getattr(sample, f'class_{i}'))
                                for i in range(self.num_classes)],
                               [[self.vocab[token] for token in getattr(sample, f'sent_{i}')]
                                for i in range(self.num_sentences)])
                              for sample in raw_valid_set]

        self.test_dataset = [(int(getattr(sample, 'entity')),
                              [int(getattr(sample, f'class_{i}'))
                               for i in range(self.num_classes)],
                              [[self.vocab[token] for token in getattr(sample, f'sent_{i}')]
                               for i in range(self.num_sentences)])
                             for sample in raw_test_set]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=generate_batch, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=generate_batch)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=generate_batch)


def generate_batch(batch: List[Tuple[int, List[int], List[List[int]]]]) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    _, classes_batch, token_lists_batch, = zip(*batch)

    tokens_batch_1, tokens_batch_2, tokens_batch_3 = zip(*token_lists_batch)

    num_tokens_batch_1 = [len(tokens) for tokens in tokens_batch_1]
    num_tokens_batch_2 = [len(tokens) for tokens in tokens_batch_2]
    num_tokens_batch_3 = [len(tokens) for tokens in tokens_batch_3]

    tokens_batch_1 = [tensor(tokens) for tokens in tokens_batch_1]
    tokens_batch_2 = [tensor(tokens) for tokens in tokens_batch_2]
    tokens_batch_3 = [tensor(tokens) for tokens in tokens_batch_3]

    tokens_batch_concated_1 = torch.cat(tokens_batch_1)
    tokens_batch_concated_2 = torch.cat(tokens_batch_2)
    tokens_batch_concated_3 = torch.cat(tokens_batch_3)

    offset_batch_1 = torch.tensor([0] + num_tokens_batch_1[:-1]).cumsum(dim=0)
    offset_batch_2 = torch.tensor([0] + num_tokens_batch_2[:-1]).cumsum(dim=0)
    offset_batch_3 = torch.tensor([0] + num_tokens_batch_3[:-1]).cumsum(dim=0)

    return tokens_batch_concated_1, offset_batch_1, \
           tokens_batch_concated_2, offset_batch_2, \
           tokens_batch_concated_3, offset_batch_3, \
           tensor(classes_batch)
