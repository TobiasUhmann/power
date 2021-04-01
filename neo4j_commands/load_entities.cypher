// Assert unique entity IDs
CREATE CONSTRAINT UniqueEntityId ON (e:Entity) ASSERT e.id IS UNIQUE;

// Load entities
LOAD CSV WITH HEADERS FROM 'file:///entities.tsv' AS row FIELDTERMINATOR '\t'
MERGE (ent:Entity {id: toInteger(row.ent), label: row.ent_lbl})
RETURN COUNT(*);

MATCH (ent:Entity)
RETURN ent
LIMIT 5;

MATCH (n) DETACH DELETE n;
