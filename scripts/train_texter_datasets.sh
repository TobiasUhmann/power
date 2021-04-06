#!/bin/bash

PYTHONPATH=src/ \
nohup python src/train_texter_datasets.py \
> logs/train_texter_datasets_$(date +'%Y-%m-%d_%H-%M-%S').stdout &
