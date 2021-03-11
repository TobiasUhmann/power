"""
The `OWER Triples DB` contains the triples in a queryable database. It is built as
an intermediate step while building the `OWER Directory` and kept for debugging
purposes.

**Structure**

::

    CREATE TABLE triples (
        head    INT,
        rel     INT,
        tail    INT
    )

::

    CREATE INDEX head_index
    ON triples(head)

::

    CREATE INDEX rel_index
    ON triples(rel)

::

    CREATE INDEX tail_index
    ON triples(tail)

|
"""

from pathlib import Path
from sqlite3 import connect
from typing import List, Tuple, Set

from dao.base_file import BaseFile


class TriplesDb(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    def create_triples_table(self) -> None:
        create_table_sql = '''
            CREATE TABLE triples (
                head    INT,
                rel     INT,
                tail    INT
            )
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

        with connect(self._path) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_head_index_sql)
            cursor.execute(create_rel_index_sql)
            cursor.execute(create_tail_index_sql)
            cursor.close()

    def insert_triples(self, triples: List[Tuple[int, int, int]]) -> None:
        sql = '''
            INSERT INTO triples (head, rel, tail)
            VALUES (?, ?, ?)
        '''

        with connect(self._path) as conn:
            conn.executemany(sql, triples)

    def select_heads_with_rel_tail(self, rel: int, tail: int) -> Set[int]:
        sql = '''
            SELECT head
            FROM triples
            WHERE rel = ? AND tail = ?
        '''

        with connect(self._path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (rel, tail))
            rows = cursor.fetchall()
            cursor.close()

        return {row[0] for row in rows}