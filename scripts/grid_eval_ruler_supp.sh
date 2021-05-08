#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_eval_ruler_supp.py \
> logs/grid_eval_ruler_supp_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
