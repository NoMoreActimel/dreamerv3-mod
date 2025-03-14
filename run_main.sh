#!/bin/bash

# enable wandb in train script
# export WANDB_API_KEY=""
# export WANDB_MODE="offline"

OUTPUT_DIR="/scratch/ss19021/dreamerv3-mod/logs/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

python dreamerv3/main.py \
  --logdir "$OUTPUT_DIR" \
  --configs crafter \
  --run.train_ratio 32
