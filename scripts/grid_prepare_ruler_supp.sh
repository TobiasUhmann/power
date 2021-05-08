#!/bin/bash

PYTHONPATH=src/ \
nohup python src/grid_prepare_ruler_supp.py \
> logs/grid_prepare_ruler_supp_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
