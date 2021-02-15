import os
from typing import List, Tuple

from pytorch_lightning import LightningDataModule
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torchtext.data import TabularDataset, Field
from torchtext.vocab import Vocab


class DataModule(LightningDataModule):
    data_dir: str
    class_count: int
    num_sentences: int
    batch_size: int
    sent_len: int

    vocab: Vocab

    train_dataset: List[Tuple[int, List[int], List[List[int]]]]
    valid_dataset: List[Tuple[int, List[int], List[List[int]]]]
    test_dataset: List[Tuple[int, List[int], List[List[int]]]]

    def __init__(self, data_dir: str, class_count: int, sent_count: int, batch_size: int, sent_len: int):
        super().__init__()

        self.data_dir = data_dir
        self.class_count = class_count
        self.num_sentences = sent_count
        self.batch_size = batch_size
        self.sent_len = sent_len

    #
    # Prepare data
    #

    def prepare_data(self):
        train_samples_tsv = os.path.join(self.data_dir, 'train.tsv')
        valid_samples_tsv = os.path.join(self.data_dir, 'valid.tsv')
        test_samples_tsv = os.path.join(self.data_dir, 'test.tsv')

        def tokenize(text: str):
            return text.split()

        fields = [('entity', Field(sequential=False, use_vocab=False))] + \
                 [(f'class_{i}', Field(sequential=False, use_vocab=False))
                  for i in range(self.class_count)] + \
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
                                for i in range(self.class_count)],
                               [[self.vocab[token] for token in getattr(sample, f'sent_{i}')]
                                for i in range(self.num_sentences)])
                              for sample in raw_train_set]

        self.valid_dataset = [(int(getattr(sample, 'entity')),
                               [int(getattr(sample, f'class_{i}'))
                                for i in range(self.class_count)],
                               [[self.vocab[token] for token in getattr(sample, f'sent_{i}')]
                                for i in range(self.num_sentences)])
                              for sample in raw_valid_set]

        self.test_dataset = [(int(getattr(sample, 'entity')),
                              [int(getattr(sample, f'class_{i}'))
                               for i in range(self.class_count)],
                              [[self.vocab[token] for token in getattr(sample, f'sent_{i}')]
                               for i in range(self.num_sentences)])
                             for sample in raw_test_set]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.generate_batch, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=self.generate_batch)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.generate_batch)

    def generate_batch(self, batch: List[Tuple[int, List[int], List[List[int]]]]) -> Tuple[Tensor, Tensor]:
        _ent, classes_batch, sents_batch, = zip(*batch)

        cropped_sents_batch = [[sent[:self.sent_len]
                                for sent in sents] for sents in sents_batch]

        padded_sents_batch = [[sent + [0] * (self.sent_len - len(sent))
                               for sent in sents] for sents in cropped_sents_batch]

        return tensor(padded_sents_batch), tensor(classes_batch)
