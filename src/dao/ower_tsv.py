from typing import List


def write_ower_tsv(ower_tsv: str, rows: List) -> None:
    with open(ower_tsv, 'w', encoding='utf-8') as f:
        for ent, label_male, label_married, label_american, label_actor, context in rows:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'
                    .format(ent, label_male, label_married, label_american, label_actor, context.replace('"', '')))
