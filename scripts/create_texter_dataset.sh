#!/bin/bash

PYTHONPATH=src/ \
nohup python src/create_texter_dataset.py \
  data/irt/split/cde/ \
  data/irt/text/cde-irt-5-marked/ \
  data/power/samples/cde-irt-5-marked/ \
  --class-count 100 \
  --sent-count 5 \
> logs/create_texter_dataset_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
