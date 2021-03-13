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

from dataclasses import dataclass
from pathlib import Path
from sqlite3 import connect
from typing import List, Tuple, Set

from dao.base_file import BaseFile


@dataclass
class DbTriple:
    head: int
    rel: int
    tail: int


class TriplesDb(BaseFile):

    def __init__(self, name: str, path: Path):
        super().__init__(name, path)

    def create_triples_table(self) -> None:
        with connect(self._path) as conn:
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

            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_head_index_sql)
            cursor.execute(create_rel_index_sql)
            cursor.execute(create_tail_index_sql)
            cursor.close()

    def insert_triples(self, db_triples: List[DbTriple]) -> None:
        with connect(self._path) as conn:
            sql = '''
                INSERT INTO triples (head, rel, tail)
                VALUES (?, ?, ?)
            '''

            conn.executemany(sql, ((t.head, t.rel, t.tail) for t in db_triples))

    def select_entities_with_class(self, class_: Tuple[int, int]) -> Set[int]:
        with connect(self._path) as conn:
            sql = '''
                SELECT head
                FROM triples
                WHERE rel = ? AND tail = ?
            '''

            cursor = conn.cursor()
            cursor.execute(sql, (class_[0], class_[1]))
            rows = cursor.fetchall()
            cursor.close()

            return {row[0] for row in rows}
