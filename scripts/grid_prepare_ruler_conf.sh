#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_prepare_ruler_conf.py \
> logs/grid_prepare_ruler_conf_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
