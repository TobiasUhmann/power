from typing import List


def write_samples_tsv(samples_tsv: str, rows: List) -> None:
    with open(samples_tsv, 'w', encoding='utf-8') as f:
        for row in rows:
            ent = row[0]
            classes = row[1:-1]
            context = row[-1]

            f.write(str(ent))
            for class_ in classes:
                f.write(f'\t{str(class_)}')
            f.write(f'\t{context}\n')
