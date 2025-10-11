#!/usr/bin/env bash

# --- Key Change: Set and export PYTHONPATH ---
# This ensures that the path to your project is added to the Python path
# for the script and any child processes it launches.
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# --- Use the modern and more robust torchrun launcher ---
# torchrun is the recommended replacement for torch.distributed.launch
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}