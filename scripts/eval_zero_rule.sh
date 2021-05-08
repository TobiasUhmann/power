#!/bin/bash

PYTHONPATH=src/ \
nohup python src/eval_zero_rule.py \
  data/power/samples/fb-irt-5-marked/ \
  100 \
  5 \
   data/power/split/fb-50/ \
> logs/eval_zero_rule_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
