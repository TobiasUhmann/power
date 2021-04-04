#!/bin/bash

PYTHONPATH=src/ \
nohup python src/create_split.py \
  data/irt/split/fb/ \
  data/irt/text/fb-irt-5-marked/ \
  data/power/split/fb-100/ \
  --known 100 \
> logs/create_split_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
