#!/bin/bash

PYTHONPATH=src/ \
nohup python src/eval_ruler.py \
  data/power/ruler/cde-50.pkl \
  data/power/texter/cde-irt-5-marked.pkl \
  5 \
  data/power/split/cde-50/ \
  data/irt/text/cde-irt-5-marked/ \
> logs/eval_ruler_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
