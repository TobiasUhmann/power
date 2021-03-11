"""
The `Ryn Directory` contains the `Ryn Dasaset`, i.e. the triples split
as well as the sentences describing the entities.

**Structure**

::

    ryn/            # Ryn Directory

        split/      # Ryn Split Directory
        text/       # Ryn Text Directory

|
"""

from pathlib import Path

from dao.base_dir import BaseDir
from dao.ryn.split.ryn_split_dir import RynSplitDir
from dao.ryn.text.ryn_text_dir import RynTextDir


class RynDir(BaseDir):

    split_dir: RynSplitDir
    text_dir: RynTextDir

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

        self.split_dir = RynSplitDir('Ryn Split Directory', path.joinpath('../../../../../src/dao/ryn/split'))
        self.text_dir = RynTextDir('Ryn Text Directory', path.joinpath('../../../../../src/dao/ryn/text'))

    def check(self) -> None:
        super().check()

        self.split_dir.check()
        self.text_dir.check()

    def create(self) -> None:
        super().create()

        self.split_dir.create()
        self.text_dir.create()
