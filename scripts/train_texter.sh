#!/bin/bash

PYTHONPATH=src/ \
nohup python src/train_texter.py \
  data/power/samples/cde-irt-5-marked/ \
  100 \
  5 \
  data/power/split/cde-100/ \
  data/power/texter/cde-irt-5-marked.pkl \
  --epoch-count 2 \
> logs/train_texter_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
