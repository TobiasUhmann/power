from pathlib import Path

from data.base_dir import BaseDir
from data.power.classes_tsv import ClassesTsv
from data.power.model.ruler_pkl import RulerPkl
from data.power.model.texter_pkl import TexterPkl
from data.irt.split.labels_txt import LabelsTxt


class ModelDir(BaseDir):
    texter_pkl: TexterPkl
    ruler_pkl: RulerPkl
    classes_tsv: ClassesTsv
    ent_labels_txt: LabelsTxt
    rel_labels_txt: LabelsTxt

    def __init__(self, path: Path):
        super().__init__(path)

        self.texter_pkl = TexterPkl(path.joinpath('texter.pkl'))
        self.ruler_pkl = RulerPkl(path.joinpath('ruler.pkl'))
        self.classes_tsv = ClassesTsv(path.joinpath('classes.tsv'))
        self.ent_labels_txt = LabelsTxt(path.joinpath('ent_labels.txt'))
        self.rel_labels_txt = LabelsTxt(path.joinpath('rel_labels.txt'))

    def check(self) -> None:
        super().check()

        self.texter_pkl.check()
        self.ruler_pkl.check()
        self.classes_tsv.check()
        self.ent_labels_txt.check()
        self.rel_labels_txt.check()
