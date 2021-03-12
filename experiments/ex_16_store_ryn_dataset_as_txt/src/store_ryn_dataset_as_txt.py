from typing import Dict

from ryn.graphs.split import Dataset

from src.dao.oid_to_rid_txt import load_oid_to_rid

if __name__ == '__main__':
    dataset = Dataset.load(path='data/oke.fb15k237_30061990_50')
    id2rel = dataset.id2rel

    oid_to_rid: Dict[str, int] = load_oid_to_rid('data/entity2id.txt')
    rid_to_oid = {rid: oid for oid, rid in oid_to_rid.items()}

    with open('data/train.txt', 'w', encoding='utf-8') as f:
        for triple in dataset.cw_train.triples:
            f.write('{}\t{}\t{}\n'.format(
                rid_to_oid[triple[0]], id2rel[triple[2]], rid_to_oid[triple[1]]))
