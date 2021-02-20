from dataclasses import dataclass
from os import path
from typing import List, Tuple

from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vocab


@dataclass
class Sample:
    ent: int
    classes: List[int]
    sents: List[List[int]]

    def __iter__(self):
        return iter((self.ent, self.classes, self.sents))


class DataModule:
    _ower_dir: str
    _class_count: int
    _sent_count: int
    _batch_size: int
    _sent_len: int

    _train_set: List[Sample]
    _valid_set: List[Sample]
    _test_set: List[Sample]

    vocab: Vocab

    def __init__(self, ower_dir: str, class_count: int, sent_count: int, batch_size: int, sent_len: int):
        super().__init__()

        self._ower_dir = ower_dir
        self._class_count = class_count
        self._sent_count = sent_count
        self._batch_size = batch_size
        self._sent_len = sent_len

    def load_datasets(self) -> None:
        """
        Read train/valid/test Sample TSVs from OWER Directory and stores the data
        internally. Must be called before retrieving DataLoaders. Also, the vocabulary
        is built over the first sentence column of the train Samples TSV. Sentences are
        tokenized by whitespace and tokens are mapped to IDs.

        Example: "Barack Obama is male. Obama is married."
                 -> ['Barack', 'Obama', 'is', 'male.', 'Obama', 'is', 'married.']
                 -> [1, 2, 3, 4, 2, 3, 5]
                 -> self.vocab = {1: 'Barack', 2: 'Obama', 3: 'is', 4: 'male.', 5: 'married.'}
        """

        #
        # Define columns for subsequent read into TabularDatasets
        #

        def tokenize(text: str):
            return text.split()

        ent_field = ('ent', Field(sequential=False, use_vocab=False))

        class_fields = [(f'class_{i}', Field(sequential=False, use_vocab=False))
                        for i in range(self._class_count)]

        sent_fields = [(f'sent_{i}', Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True))
                       for i in range(self._sent_count)]

        fields = [ent_field] + class_fields + sent_fields

        #
        # Read Samples TSVs into TabularDatasets
        #

        train_samples_tsv = path.join(self._ower_dir, 'train.tsv')
        valid_samples_tsv = path.join(self._ower_dir, 'valid.tsv')
        test_samples_tsv = path.join(self._ower_dir, 'test.tsv')

        raw_train_set = TabularDataset(train_samples_tsv, 'tsv', fields, skip_header=True)
        raw_valid_set = TabularDataset(valid_samples_tsv, 'tsv', fields, skip_header=True)
        raw_test_set = TabularDataset(test_samples_tsv, 'tsv', fields, skip_header=True)

        #
        # Build vocab
        #

        first_sent_field = sent_fields[0][1]
        first_sent_field.build_vocab(raw_train_set)
        self.vocab = first_sent_field.vocab

        #
        # TabularDataset -> List[Sample] (i.e. parse ints, map tokens -> IDs)
        #

        def transform(raw_set: TabularDataset) -> List[Sample]:
            return [Sample(
                int(getattr(row, 'ent')),
                [int(getattr(row, f'class_{i}')) for i in range(self._class_count)],
                [[self.vocab[token] for token in getattr(row, f'sent_{i}')] for i in range(self._sent_count)]
            ) for row in raw_set]

        self._train_set = transform(raw_train_set)
        self._valid_set = transform(raw_valid_set)
        self._test_set = transform(raw_test_set)

    def get_train_loader(self) -> DataLoader:
        return DataLoader(self._train_set, batch_size=self._batch_size, collate_fn=self._generate_batch, shuffle=True)

    def get_valid_loader(self) -> DataLoader:
        return DataLoader(self._valid_set, batch_size=self._batch_size, collate_fn=self._generate_batch)

    def get_test_loader(self) -> DataLoader:
        return DataLoader(self._test_set, batch_size=self._batch_size, collate_fn=self._generate_batch)

    def _generate_batch(self, batch: List[Tuple[int, List[int], List[List[int]]]]) -> Tuple[Tensor, Tensor]:
        _ent, classes_batch, sents_batch, = zip(*batch)

        cropped_sents_batch = [[sent[:self._sent_len]
                                for sent in sents] for sents in sents_batch]

        padded_sents_batch = [[sent + [0] * (self._sent_len - len(sent))
                               for sent in sents] for sents in cropped_sents_batch]

        return tensor(padded_sents_batch), tensor(classes_batch)
