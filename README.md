# Assumed Data Directory Structure

```
data/
    anyburl/
        cde/
            rules/                      AnyBURL Rules Dir
            facts.tsv                   AnyBURL Facts TSV             
        ...

    irt/
        split/
            cde/                        IRT Split Dir (CoDEx)
            fb/                         IRT Split Dir (FB15K-237)
        text/
            cde-irt-5-marked/           IRT Text Dir (CoDEx graph, IRT sentences, 5 per entity, marked)
            ...
            
    power/
        ruler/
            cde-test-50/                POWER Ruler Dir (CoDEx graph, test data, 50% known test triples)
                rules/                  POWER Rules Dir
                ruler.pkl               POWER Ruler PKL
            ...
        split/
            cde-50/                     POWER Split Dir (CoDEx graph, 50% known valid/test triples)
            ...
        texter/
            cde-irt-5-marked.pkl        POWER Texter PKL (CoDEx graph, IRT sentences, 5 per entity, marked)
            ...
```