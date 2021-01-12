from typing import Dict


def read_mid2rid_txt(path: str) -> Dict[str, int]:
    """
    :param path: path to mid2rid TXT
    :return: dict: Freebase MID -> ryn ID
    """

    mid2rid = dict()

    with open(path) as fh:
        next(fh)
        for line in fh.readlines():
            mid, rid = line.split('\t')
            mid2rid[mid] = int(rid)

    return mid2rid
