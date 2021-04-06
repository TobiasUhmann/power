#!/bin/bash

PYTHONPATH=src/ \
nohup python src/create_anyburl_dataset.py \
  data/power/split/cde-50/ \
  data/anyburl/cde/facts.tsv \
> logs/create_anyburl_dataset_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
