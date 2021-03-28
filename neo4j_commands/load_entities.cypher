CREATE CONSTRAINT UniqueEntityId ON (e:Entity) ASSERT e.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///entities.tsv' AS row FIELDTERMINATOR '\t'
WITH toInteger(row.ent) AS ent, row.ent_lbl AS ent_lbl
MERGE (e:Entity {id: ent, label: ent_lbl})
RETURN COUNT(*);

MATCH (e:Entity)
RETURN e
LIMIT 5;

MATCH (n) DETACH DELETE n;
