from typing import Dict


def read_oid_to_rid_txt(oid_to_rid_txt: str) -> Dict[str, int]:
    """
    :param oid_to_rid_txt: path to OID-to-RID TXT
    :return: dict: OID -> RID
    """

    oid_to_rid = dict()

    with open(oid_to_rid_txt) as f:
        next(f)
        for line in f.readlines():
            oid, rid = line.split('\t')
            oid_to_rid[oid] = int(rid)

    return oid_to_rid
