from typing import Callable, List

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler
from torchnlp.encoders import LabelEncoder


class BertDataModule(pl.LightningDataModule):
    batch_size: int
    collate_fn: Callable
    loader_workers: int
    test_csv: str
    train_csv: str
    valid_csv: str

    def __init__(self,
                 batch_size: int,
                 collate_fn: Callable,
                 loader_workers: int,
                 test_csv: str,
                 train_csv: str,
                 valid_csv: str,
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.loader_workers = loader_workers
        self.test_csv = test_csv
        self.train_csv = train_csv
        self.valid_csv = valid_csv

        #
        # Set up label encoder
        #

        self.label_encoder = LabelEncoder(
            pd.read_csv(self.train_csv).label.astype(str).unique().tolist(),
            reserved_labels=[],
        )

        self.label_encoder.unknown_index = None

    def train_dataloader(self) -> DataLoader:
        train_dataset = read_csv(self.train_csv)

        return DataLoader(
            dataset=train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = read_csv(self.valid_csv)

        return DataLoader(
            dataset=valid_dataset,
            sampler=RandomSampler(valid_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset = read_csv(self.test_csv)

        return DataLoader(
            dataset=test_dataset,
            sampler=RandomSampler(test_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.loader_workers,
        )


def read_csv(path: str) -> List:
    df = pd.read_csv(path)
    df = df[['description', 'is_married', 'is_male', 'is_american', 'is_actor']]

    df['description'] = df['description'].astype(str)
    df['is_married'] = df['is_married'].astype(str)
    df['is_male'] = df['is_male'].astype(str)
    df['is_american'] = df['is_american'].astype(str)
    df['is_actor'] = df['is_actor'].astype(str)

    return df.to_dict('records')
