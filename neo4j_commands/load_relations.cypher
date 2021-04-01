// Load CW train facts
LOAD CSV WITH HEADERS FROM 'file:///cw_train_facts.tsv' AS row FIELDTERMINATOR '\t'
MATCH (head:Entity {id: toInteger(row.head)})
MATCH (tail:Entity {id: toInteger(row.tail)})
CALL apoc.merge.relationship(
  head,
  'R_' + row.rel,
  {id: toInteger(row.rel), label: row.rel_lbl, type: 'cw_train'},
  {},
  tail
) YIELD rel
RETURN COUNT(*);

// Show some entities and relations
MATCH (h)-[r]->(t)
RETURN h, r, t LIMIT 25;
