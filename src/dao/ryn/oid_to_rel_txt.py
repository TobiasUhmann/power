from typing import Dict


def load(oid_to_rid_txt: str) -> Dict[str, int]:
    """
    :param oid_to_rid_txt: path to OID-to-RID TXT
    :return: OID -> RID mapping
    """

    with open(oid_to_rid_txt) as f:
        lines = f.readlines()

    oid_to_rid: Dict[str, int] = {}

    for line in lines[1:]:
        parts = line.split()

        label = ' '.join(parts[:-1])
        rid = int(parts[-1])

        oid_to_rid[label] = rid

    return oid_to_rid
