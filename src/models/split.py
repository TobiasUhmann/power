from enum import Enum


class Split(Enum):
    cw_train = 1
    cw_valid = 2
    ow_valid = 3
    ow_test = 4