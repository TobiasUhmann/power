#!/bin/bash

PYTHONPATH=src/ \
nohup python src/eval_ruler.py \
  data/power/texter/cde-cde-5-clean.pkl \
  5 \
  data/power/split/cde-0/ \
  data/irt/text/cde-cde-5-clean/ \
> logs/eval_ruler_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
