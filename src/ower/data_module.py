import os
from typing import List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchtext.data import TabularDataset, Field
from torchtext.vocab import Vocab


class DataModule(LightningDataModule):
    data_dir: str
    batch_size: int

    vocab: Vocab
    num_classes: int

    train_dataset: List[Tuple[int, List[int], List[int]]]
    valid_dataset: List[Tuple[int, List[int], List[int]]]
    test_dataset: List[Tuple[int, List[int], List[int]]]

    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    #
    # Prepare data
    #

    def prepare_data(self):
        train_samples_tsv = os.path.join(self.data_dir, 'samples-v1-train.tsv')
        valid_samples_tsv = os.path.join(self.data_dir, 'samples-v1-valid.tsv')
        test_samples_tsv = os.path.join(self.data_dir, 'samples-v1-test.tsv')

        with open(train_samples_tsv, encoding='utf-8') as f:
            train_header = f.readline()
            train_header_len = len(train_header.split('\t'))

        with open(valid_samples_tsv, encoding='utf-8') as f:
            valid_header = f.readline()
            valid_header_len = len(valid_header.split('\t'))

        with open(test_samples_tsv, encoding='utf-8') as f:
            test_header = f.readline()
            test_header_len = len(test_header.split('\t'))

        assert train_header_len == valid_header_len == test_header_len

        self.num_classes = train_header_len - 2

        def tokenize(text: str):
            return text.split()

        fields = [('entity', Field())] + \
                 [(str(i), Field()) for i in range(self.num_classes)] + \
                 [('context', Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True))]

        raw_train_set = TabularDataset(train_samples_tsv, 'tsv', fields, skip_header=True)
        raw_valid_set = TabularDataset(valid_samples_tsv, 'tsv', fields, skip_header=True)
        raw_test_set = TabularDataset(test_samples_tsv, 'tsv', fields, skip_header=True)

        context_field = fields[-1][1]
        context_field.build_vocab(raw_train_set)
        self.vocab = context_field.vocab

        self.train_dataset = [(int(getattr(sample, 'entity')),
                               [int(getattr(sample, str(i))) for i in range(self.num_classes)],
                               [self.vocab[token] for token in sample.context])
                              for sample in raw_train_set]

        self.valid_dataset = [(int(getattr(sample, 'entity')),
                               [int(getattr(sample, str(i))) for i in range(self.num_classes)],
                               [self.vocab[token] for token in sample.context])
                              for sample in raw_valid_set]

        self.test_dataset = [(int(getattr(sample, 'entity')),
                              [int(getattr(sample, str(i))) for i in range(self.num_classes)],
                              [self.vocab[token] for token in sample.context])
                             for sample in raw_test_set]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=generate_batch)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=generate_batch)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=generate_batch)


def generate_batch(batch):
    ent = torch.tensor([entry[0] for entry in batch])
    label = torch.tensor([entry[1] for entry in batch])
    text = [entry[2] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)

    return ent, text, offsets, label
