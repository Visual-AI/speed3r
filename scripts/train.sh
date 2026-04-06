#!/bin/bash

# Training script for Speed3r
# Usage: bash scripts/train.sh

HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 --num_machines 1 scripts/train_pi3_sparse.py \
    train=train_pi3_sparse \
    name=experiment_01
