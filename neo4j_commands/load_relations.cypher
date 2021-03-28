// Load CW train facts
LOAD CSV WITH HEADERS FROM "file:///cw_train_facts.tsv" AS row FIELDTERMINATOR '\t'
WITH toInteger(row.head) AS head, row.rel AS rel_, row.rel_lbl AS rel_lbl, toInteger(row.tail) AS tail
MATCH (h:Entity) WHERE h.id = head
MATCH (t:Entity) WHERE t.id = tail
CALL apoc.merge.relationship(h, "R_" + rel_, {label: rel_lbl, type: "cw_train"}, {}, t) YIELD rel
RETURN COUNT(*);

// Load CW valid facts
LOAD CSV WITH HEADERS FROM "file:///cw_valid_facts.tsv" AS row FIELDTERMINATOR '\t'
WITH toInteger(row.head) AS head, row.rel AS rel_, row.rel_lbl AS rel_lbl, toInteger(row.tail) AS tail
MATCH (h:Entity) WHERE h.id = head
MATCH (t:Entity) WHERE t.id = tail
CALL apoc.merge.relationship(h, "R_" + rel_, {label: rel_lbl, type: "cw_valid"}, {}, t) YIELD rel
RETURN COUNT(*);

// Load OW valid facts
LOAD CSV WITH HEADERS FROM "file:///ow_valid_facts.tsv" AS row FIELDTERMINATOR '\t'
WITH toInteger(row.head) AS head, row.rel AS rel_, row.rel_lbl AS rel_lbl, toInteger(row.tail) AS tail
MATCH (h:Entity) WHERE h.id = head
MATCH (t:Entity) WHERE t.id = tail
CALL apoc.merge.relationship(h, "R_" + rel_, {label: rel_lbl, type: "ow_valid"}, {}, t) YIELD rel
RETURN COUNT(*);

// Load OW test facts
LOAD CSV WITH HEADERS FROM "file:///ow_test_facts.tsv" AS row FIELDTERMINATOR '\t'
WITH toInteger(row.head) AS head, row.rel AS rel_, row.rel_lbl AS rel_lbl, toInteger(row.tail) AS tail
MATCH (h:Entity) WHERE h.id = head
MATCH (t:Entity) WHERE t.id = tail
CALL apoc.merge.relationship(h, "R_" + rel_, {label: rel_lbl, type: "ow_test"}, {}, t) YIELD rel
RETURN COUNT(*);

// Show some entities and relations
MATCH (h)-[r]->(t)
RETURN h, r, t
LIMIT 25;
