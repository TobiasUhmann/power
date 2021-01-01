from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchtext.data import TabularDataset, Field, Dataset


class DataModule(LightningDataModule):
    data_dir: str
    batch_size: int

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

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
                                                                train='train_outputs.tsv',
                                                                test='test_outputs.tsv',
                                                                format='tsv',
                                                                skip_header=True,
                                                                fields=fields)

        train_val_len = len(train_val_dataset)
        train_len = int(train_val_len * 0.95)
        val_len = train_val_len - train_len

        train_dataset, val_dataset = random_split(train_val_dataset, [train_len, val_len])

        #
        # Store datasets
        #

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
