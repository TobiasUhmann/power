"""
The `Ryn Split Directory` contains the files that define the triples
split into train / valid / test.

**Structure**

::

    split/                  # Ryn Split Directory

        entity2id.txt       # Ryn Labels TXT
        relation2id.txt     # Ryn Labels TXT

        cw.train2id.txt     # Ryn Triples TXT
        cw.valid2id.txt     # Ryn Triples TXT
        ow.valid2id.txt     # Ryn Triples TXT
        ow.test2id.txt      # Ryn Triples TXT

|
"""

from pathlib import Path

from dao.base_dir import BaseDir
from dao.ryn.split.ryn_labels_txt import RynLabelsTxt
from dao.ryn.split.ryn_triples_txt import RynTriplesTxt


class RynSplitDir(BaseDir):

    _entity_labels_txt: RynLabelsTxt
    _relation_labels_txt: RynLabelsTxt
    
    _cw_train_triples_txt: RynTriplesTxt
    _cw_valid_triples_txt: RynTriplesTxt
    _ow_valid_triples_txt: RynTriplesTxt
    _ow_test_triples_txt: RynTriplesTxt

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

        self._entity_labels_txt = RynLabelsTxt('Ryn Entity Labels TXT', path.joinpath('entity2id.txt'))
        self._relation_labels_txt = RynLabelsTxt('Ryn Relation Labels TXT', path.joinpath('relation2id.txt'))
        
        self._cw_train_triples_txt = RynTriplesTxt('Ryn CW Train Triples TXT', path.joinpath('cw.train2id.txt'))
        self._cw_valid_triples_txt = RynTriplesTxt('Ryn CW Valid Triples TXT', path.joinpath('cw.valid2id.txt'))
        self._ow_valid_triples_txt = RynTriplesTxt('Ryn OW Valid Triples TXT', path.joinpath('ow.valid2id.txt'))
        self._ow_test_triples_txt = RynTriplesTxt('Ryn OW Test Triples TXT', path.joinpath('ow.test2id.txt'))

    def check(self) -> None:
        super().check()
            
        self._entity_labels_txt.check()
        self._relation_labels_txt.check()

        self._cw_train_triples_txt.check()
        self._cw_valid_triples_txt.check()
        self._ow_valid_triples_txt.check()
        self._ow_test_triples_txt.check()
