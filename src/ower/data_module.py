from typing import Optional, List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchtext.data import TabularDataset, Field


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


class DataModule(LightningDataModule):
    data_dir: str
    batch_size: int

    train_dataset: List[Tuple[List[float], torch.Tensor]]
    val_dataset: List[Tuple[List[float], torch.Tensor]]
    test_dataset: List[Tuple[List[float], torch.Tensor]]

    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        #
        # Read dataset TSV
        #

        tokenize = lambda x: x.split()

        is_male_field = Field(sequential=False, use_vocab=False)
        is_married_field = Field(sequential=False, use_vocab=False)
        is_american_field = Field(sequential=False, use_vocab=False)
        is_actor_field = Field(sequential=False, use_vocab=False)
        context_field = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)

        fields = [('entity', None),
                  ('is_male', is_male_field),
                  ('is_married', is_married_field),
                  ('is_american', is_american_field),
                  ('is_actor', is_actor_field),
                  ('context', context_field)]

        #
        # Split full dataset into train/val/test
        #

        train_val_dataset, test_dataset = TabularDataset.splits(path=self.data_dir,
                                                                train='samples-v1-train.tsv',
                                                                test='samples-v1-test.tsv',
                                                                format='tsv',
                                                                skip_header=True,
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
            [float(x.is_male), float(x.is_married), float(x.is_american), float(x.is_actor)],
            torch.tensor([vocab[t] for t in x.context])
        ) for x in train_dataset]

        transformed_val_dataset = [(
            [float(x.is_male), float(x.is_married), float(x.is_american), float(x.is_actor)],
            torch.tensor([vocab[t] for t in x.context])
        ) for x in val_dataset]

        transformed_test_dataset = [(
            [float(x.is_male), float(x.is_married), float(x.is_american), float(x.is_actor)],
            torch.tensor([vocab[t] for t in x.context])
        ) for x in test_dataset]

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
