"""
Load the triples from the OpenKE train/valid/test TXTs into an Sqlite DB for
easier exploration. The Freebase MIDs used in the OpenKE dataset are translated
to Wikidata labels.

Example rows in output triples DB::

    train    Dominican Republic    /location/country/form_of_government    republic
    train    Mighty Morphin ...    /tv/tv_program/regular_cast./tv/r...    Wendee Lee

|
"""

import json
import sqlite3

from tqdm import tqdm


def main():
    with open('data/entity2wikidata.json') as f:
        entity2wikidata = json.load(f)

    with sqlite3.connect('data/triples.db') as conn:
        create_triples_table(conn)

        process_txt('data/FB15K-237.2/Release/train.txt', 'train', entity2wikidata, conn)
        process_txt('data/FB15K-237.2/Release/valid.txt', 'valid', entity2wikidata, conn)
        process_txt('data/FB15K-237.2/Release/test.txt', 'test', entity2wikidata, conn)


def create_triples_table(conn):
    create_table_sql = '''
        CREATE TABLE triples (
            dataset TEXT,
            head TEXT,
            rel TEXT,
            tail TEXT
        )
    '''

    create_dataset_index_sql = '''
        CREATE INDEX dataset_index
        ON triples(dataset)
    '''

    create_head_index_sql = '''
        CREATE INDEX head_index
        ON triples(head)
    '''

    create_rel_index_sql = '''
        CREATE INDEX rel_index
        ON triples(rel)
    '''

    create_tail_index_sql = '''
        CREATE INDEX tail_index
        ON triples(tail)
    '''

    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    cursor.execute(create_dataset_index_sql)
    cursor.execute(create_head_index_sql)
    cursor.execute(create_rel_index_sql)
    cursor.execute(create_tail_index_sql)
    cursor.close()


def insert_triple(conn, triple):
    dataset, head, rel, tail = triple

    sql = '''
        INSERT INTO triples (dataset, head, rel, tail)
        VALUES (?, ?, ?, ?)
    '''

    cursor = conn.cursor()
    cursor.execute(sql, (dataset, head, rel, tail))
    cursor.close()


def process_txt(file, dataset, entity2wikidata, conn):
    with open(file) as f:
        for line in tqdm(f):
            head, rel, tail = line.split('\t')
            head, rel, tail = head.strip(), rel.strip(), tail.strip()

            head_lbl = entity2wikidata[head]['label'] if head in entity2wikidata else head
            tail_lbl = entity2wikidata[tail]['label'] if tail in entity2wikidata else tail

            insert_triple(conn, (dataset, head_lbl, rel, tail_lbl))


if __name__ == '__main__':
    main()
