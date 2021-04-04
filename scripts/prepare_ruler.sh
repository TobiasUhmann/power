#!/bin/bash

PYTHONPATH=src/ \
nohup python src/prepare_ruler.py \
  data/power/ruler/cde-50/rules.tsv \
  bolt://localhost:7687 \
  neo4j \
  1234567890 \
  data/power/split/cde-50/ \
  data/power/ruler/cde-50/ruler.pkl \
> logs/prepare_ruler_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
