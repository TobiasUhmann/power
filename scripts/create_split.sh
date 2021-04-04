#!/bin/bash

PYTHONPATH=src/ \
nohup python src/create_split.py \
  data/irt/split/cde/ \
  data/irt/text/cde-irt-5-marked/ \
  data/power/split/cde-50/ \
  --known 50 \
> logs/create_split_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
