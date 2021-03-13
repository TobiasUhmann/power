"""
The `OWER Directory` contains the input files required for training the
`OWER Classifier`.

**Structure**

::

    ower/           # OWER Directory

        test.tsv    # OWER Samples TSV
        train.tsv   # OWER Samples TSV
        valid.tsv   # OWER Samples TSV

        test.db     # OWER Triples DB
        train.db    # OWER Triples DB
        valid.db    # OWER Triples DB

|
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vocab

from dao.base_dir import BaseDir
from dao.ower.ower_samples_tsv import SamplesTsv
from dao.ower.ower_triples_db import TriplesDb


@dataclass
class Sample:
    ent: int
    classes: List[int]
    sents: List[List[int]]

    def __iter__(self):
        return iter((self.ent, self.classes, self.sents))


class OwerDir(BaseDir):
    train_triples_db: TriplesDb
    valid_triples_db: TriplesDb
    test_triples_db: TriplesDb

    train_samples_tsv: SamplesTsv
    valid_samples_tsv: SamplesTsv
    test_samples_tsv: SamplesTsv

    _class_count: int
    _sent_count: int

    def __init__(self, name: str, path: Path, class_count: int, sent_count: int):
        super().__init__(name, path)

        self._class_count = class_count
        self._sent_count = sent_count

        self.train_triples_db = TriplesDb('OWER Train Triples DB', path.joinpath('train.db'))
        self.valid_triples_db = TriplesDb('OWER Valid Triples DB', path.joinpath('valid.db'))
        self.test_triples_db = TriplesDb('OWER Test Triples DB', path.joinpath('test.db'))

        self.train_samples_tsv = SamplesTsv('OWER Train Samples TSV', path.joinpath('train.tsv'))
        self.valid_samples_tsv = SamplesTsv('OWER Valid Samples TSV', path.joinpath('valid.tsv'))
        self.test_samples_tsv = SamplesTsv('OWER Test Samples TSV', path.joinpath('test.tsv'))

    def check(self) -> None:
        super().check()

        self.train_triples_db.check()
        self.valid_triples_db.check()
        self.test_triples_db.check()

        self.train_samples_tsv.check()
        self.valid_samples_tsv.check()
        self.test_samples_tsv.check()

    def read_datasets(self, vectors=None) -> Tuple[List[Sample], List[Sample], List[Sample], Vocab]:
        """
        :param vectors: Pre-trained word embeddings
        """

        def _tokenize(text: str) -> List[str]:
            return text.split()

        ent_field = Field(sequential=False, use_vocab=False)
        class_field = Field(sequential=False, use_vocab=False)
        sent_field = Field(sequential=True, use_vocab=True, tokenize=_tokenize, lower=True)

        ent_col = ('ent', ent_field)
        class_cols = [(f'class_{i}', class_field) for i in range(self._class_count)]
        sent_cols = [(f'sent_{i}', sent_field) for i in range(self._sent_count)]

        cols = [ent_col] + class_cols + sent_cols

        train_tab_set = TabularDataset(str(self.train_samples_tsv._path), 'tsv', cols, skip_header=True)
        valid_tab_set = TabularDataset(str(self.valid_samples_tsv._path), 'tsv', cols, skip_header=True)
        test_tab_set = TabularDataset(str(self.test_samples_tsv._path), 'tsv', cols, skip_header=True)

        #
        # Build vocab on train data
        #

        sent_field.build_vocab(train_tab_set, vectors=vectors)
        vocab = sent_field.vocab

        #
        # Transform TabularDataset -> List[Sample]
        #

        def _transform(raw_set: TabularDataset) -> List[Sample]:
            return [Sample(
                int(getattr(row, 'ent')),
                [int(getattr(row, f'class_{i}')) for i in range(self._class_count)],
                [[vocab[token] for token in getattr(row, f'sent_{i}')] for i in range(self._sent_count)]
            ) for row in raw_set]

        train_set = _transform(train_tab_set)
        valid_set = _transform(valid_tab_set)
        test_set = _transform(test_tab_set)

        return train_set, valid_set, test_set, vocab
