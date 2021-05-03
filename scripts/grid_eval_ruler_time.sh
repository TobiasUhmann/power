#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_eval_ruler_time.py \
> logs/grid_eval_ruler_time_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
