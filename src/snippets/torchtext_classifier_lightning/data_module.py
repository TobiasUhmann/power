import os
from typing import List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor, tensor
from torch.utils.data import DataLoader, random_split
from torchtext.data import Dataset
from torchtext.datasets import text_classification
from torchtext.vocab import Vocab


class DataModule(LightningDataModule):
    data_dir: str
    batch_size: int
    ngrams: int

    vocab: Vocab
    class_count: int

    train_dataset: Dataset
    valid_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, data_dir: str, batch_size: int, ngrams: int):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.ngrams = ngrams

    #
    # Prepare data
    #

    def prepare_data(self):
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        ag_news = text_classification.DATASETS['AG_NEWS']
        train_valid_set, test_set = ag_news(root='data/', ngrams=self.ngrams, vocab=None)

        self.vocab = train_valid_set.get_vocab()
        self.class_count = len(train_valid_set.get_labels())

        train_len = int(len(train_valid_set) * 0.7)
        valid_len = len(train_valid_set) - train_len
        train_set, valid_set = random_split(train_valid_set, [train_len, valid_len])

        self.train_dataset = train_set
        self.valid_dataset = valid_set
        self.test_dataset = test_set

    #
    # DataLoader methods
    #

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=generate_batch, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=generate_batch)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=generate_batch)


def generate_batch(label_tokens_batch: List[Tuple[int, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Split (label, tokens) batch and transform tokens into EmbeddingBag format.

    :return: 1. Concated tokens of all texts, Tensor[]
             2. Token offsets where texts begin, Tensor[batch_size]
             3. Labels for texts, Tensor[batch_size]
    """

    label_batch = tensor([entry[0] for entry in label_tokens_batch])
    tokens_batch = [entry[1] for entry in label_tokens_batch]

    num_tokens_batch = [len(tokens) for tokens in tokens_batch]

    tokens_batch_concated = torch.cat(tokens_batch)
    offset_batch = tensor([0] + num_tokens_batch[:-1]).cumsum(dim=0)

    return tokens_batch_concated, offset_batch, label_batch
