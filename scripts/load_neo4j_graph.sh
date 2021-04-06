#!/bin/bash

PYTHONPATH=src/ \
nohup python src/load_neo4j_graph.py \
  bolt://localhost:7687 \
  neo4j \
  1234567890 \
> logs/load_neo4j_graph_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
