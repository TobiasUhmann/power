#!/bin/bash

PYTHONPATH=src/ \
nohup python src/train_aggregator.py \
  data/power/ruler/cde-50.pkl \
  data/power/texter/cde-irt-5-marked.pkl \
  5 \
  data/power/split/cde-50/ \
  data/irt/text/cde-irt-5-marked/ \
  --lr 0.001 \
> logs/train_aggregator_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
