"""
The `POWER Directory` contains the input files required for training the
`POWER Classifier`. The `POWER Temp Directory` keeps intermediate files
for debugging purposes.

**Structure**

::

    power/                # POWER Directory

        tmp/              # POWER Temp Directory

        ent_labels.txt    # POWER Entity Labels TXT
        rel_labels.txt    # POWER Relation Labels TXT

        classes.tsv       # POWER Classes TSV

        test.tsv          # POWER Test Samples TSV
        train.tsv         # POWER Train Samples TSV
        valid.tsv         # POWER Valid Samples TSV

|
"""

from pathlib import Path

from data.base_dir import BaseDir
from data.power.classes_tsv import ClassesTsv
from data.power.samples_tsv import SamplesTsv
from data.power.tmp.tmp_dir import TmpDir
from data.ryn.split.labels_txt import LabelsTxt


class PowerDir(BaseDir):
    tmp_dir: TmpDir

    ent_labels_txt: LabelsTxt
    rel_labels_txt: LabelsTxt

    classes_tsv: ClassesTsv

    train_samples_tsv: SamplesTsv
    valid_samples_tsv: SamplesTsv
    test_samples_tsv: SamplesTsv

    def __init__(self, path: Path):
        super().__init__(path)

        self.tmp_dir = TmpDir(path.joinpath('tmp'))

        self.ent_labels_txt = LabelsTxt(path.joinpath('ent_labels.txt'))
        self.rel_labels_txt = LabelsTxt(path.joinpath('rel_labels.txt'))

        self.classes_tsv = ClassesTsv(path.joinpath('classes.tsv'))

        self.train_samples_tsv = SamplesTsv(path.joinpath('train.tsv'))
        self.valid_samples_tsv = SamplesTsv(path.joinpath('valid.tsv'))
        self.test_samples_tsv = SamplesTsv(path.joinpath('test.tsv'))

    def check(self) -> None:
        super().check()

        self.tmp_dir.check()

        self.ent_labels_txt.check()
        self.rel_labels_txt.check()

        self.classes_tsv.check()

        self.train_samples_tsv.check()
        self.valid_samples_tsv.check()
        self.test_samples_tsv.check()

    def create(self, overwrite=False) -> None:
        super().create(overwrite=overwrite)

        self.tmp_dir.create()
