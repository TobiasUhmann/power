from typing import List, Tuple


def save_outputs(output_txt: str, outputs: List[Tuple[int, int, int, int, int, str]]) -> None:

    with open(output_txt, 'w', encoding='utf-8') as f:
        for ent, label_male, label_married, label_american, label_actor, context in outputs:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(ent,
                                                      label_male,
                                                      label_married,
                                                      label_american,
                                                      label_actor,
                                                      context.replace('"', '')))
