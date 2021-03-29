from pathlib import Path

from data.base_dir import BaseDir
from data.power.model.ruler_pkl import RulerPkl


class ModelDir(BaseDir):
    ruler_pkl: RulerPkl

    def __init__(self, path: Path):
        super().__init__(path)

        self.ruler_pkl = RulerPkl(path.joinpath('ruler.pkl'))

    def check(self) -> None:
        super().check()

        self.ruler_pkl.check()
