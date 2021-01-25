from os import path
from typing import Optional, List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchtext.data import TabularDataset, Field


def generate_batch(batch):
    ent = torch.tensor([entry[0] for entry in batch])
    label = torch.tensor([entry[1] for entry in batch])
    text = [entry[2] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)

    return ent, text, offsets, label


class SimpleDataModule(LightningDataModule):
    data_dir: str
    batch_size: int

    train_dataset: List[Tuple[List[float], torch.Tensor]]
    val_dataset: List[Tuple[List[float], torch.Tensor]]
    test_dataset: List[Tuple[List[float], torch.Tensor]]

    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes: int

    def setup(self, stage: Optional[str] = None):
        #
        # Read dataset TSV
        #

        samples_train_tsv = path.join(self.data_dir, 'samples-v1-train.tsv')
        with open(samples_train_tsv, encoding='utf-8') as f:
            header = f.readline()

        header_length = len(header.split('\t'))
        self.num_classes = header_length - 2

        fields = [('entity', Field(sequential=False, use_vocab=False))]

        for i in range(header_length - 2):
            fields.append((str(i), Field(sequential=False, use_vocab=False)))

        context_field = Field(sequential=True,
                              use_vocab=True,
                              tokenize=lambda x: x.split(),
                              lower=True)

        fields.append(('context', context_field))

        #
        # Split full dataset into train/val/test
        #

        train_val_dataset, test_dataset = TabularDataset.splits(path=self.data_dir,
                                                                train='samples-v1-train.tsv',
                                                                test='samples-v1-test.tsv',
                                                                format='tsv',
                                                                skip_header=False,
                                                                fields=fields)

        train_val_len = len(train_val_dataset)
        train_len = int(train_val_len * 0.95)
        val_len = train_val_len - train_len

        train_dataset, val_dataset, = random_split(train_val_dataset, [train_len, val_len])

        #
        # Transform datasets
        #

        context_field.build_vocab(train_val_dataset)
        vocab = context_field.vocab

        transformed_train_dataset = [(
            float(getattr(sample, 'entity')),
            [float(getattr(sample, str(i))) for i in range(header_length - 2)],
            torch.tensor([vocab[t] for t in sample.context])
        ) for sample in train_dataset]

        transformed_val_dataset = [(
            float(getattr(sample, 'entity')),
            [float(getattr(sample, str(i))) for i in range(header_length - 2)],
            torch.tensor([vocab[t] for t in sample.context])
        ) for sample in val_dataset]

        transformed_test_dataset = [(
            float(getattr(sample, 'entity')),
            [float(getattr(sample, str(i))) for i in range(header_length - 2)],
            torch.tensor([vocab[t] for t in sample.context])
        ) for sample in test_dataset]

        #
        # Store datasets
        #

        self.train_dataset = transformed_train_dataset
        self.val_dataset = transformed_val_dataset
        self.test_dataset = transformed_test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=generate_batch)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=generate_batch)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=generate_batch)
