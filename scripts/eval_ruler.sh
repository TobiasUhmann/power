#!/bin/bash

PYTHONPATH=src/ \
nohup python src/eval_ruler.py \
  data/power/ruler/cde-50.pkl \
  data/power/split/cde-50/ \
> logs/eval_ruler_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
