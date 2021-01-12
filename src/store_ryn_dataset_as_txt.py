from typing import Dict

from ryn.graphs.split import Dataset

from src.dao.mid2rid_txt import read_mid2rid_txt

if __name__ == '__main__':
    dataset = Dataset.load(path='data/oke.fb15k237_30061990_50')
    id2rel = dataset.id2rel

    mid2rid: Dict[str, int] = read_mid2rid_txt('data/entity2id.txt')
    rid2mid = {rid: mid for mid, rid in mid2rid.items()}

    with open('data/train.txt', 'w', encoding='utf-8') as train_fh:
        for triple in dataset.cw_train.triples:
            train_fh.write('{}\t{}\t{}\n'.format(
                rid2mid[triple[0]], id2rel[triple[2]], rid2mid[triple[1]]))
