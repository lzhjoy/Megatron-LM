#!/bin/bash

cd /volume/ailab4sci/txie/huyiwen/Ubiquant-Pretrain

export CUDA_DEVICE_MAX_CONNECTIONS=1

SERVE_PORT=8001

# Evaluation Arguments
CHECKPOINT=${CHECKPOINT:-${1:-"NOT_EXISTS"}}
LOAD_ITER=${LOAD_ITER:-${2:-"LATEST"}}
BATCH_SIZE=${BATCH_SIZE:-1}
MP_SIZE=${MP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}

# Multi-node Arguments
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${PET_NNODES:-"1"}
NODE_RANK=${PET_NODE_RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


# ###################################################
# ################# Process Arguments
# ###################################################

DP_SIZE=$((WORLD_SIZE / (MP_SIZE * PP_SIZE)))
if [ $DP_SIZE -ne 1 ]; then
    echo "Error: DP_SIZE is not 1. Please check parallel dimensions."
    exit 1
fi

if [[ -e "${CHECKPOINT}/checkpoint/latest_checkpointed_iteration.txt" ]]; then
    CHECKPOINT=$CHECKPOINT/checkpoint
fi

if [[ ! -e "${CHECKPOINT}/latest_checkpointed_iteration.txt" ]]; then
    echo "Error: Checkpoint directory ${CHECKPOINT} not recognized"
    exit 1
fi

(echo >/dev/tcp/localhost/$SERVE_PORT) 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Port $SERVE_PORT is in use"
    exit 1
fi

EXTRA_ARGS=""

if [ "$LOAD_ITER" != "LATEST" ]; then
    EXTRA_ARGS=" --ckpt-step $LOAD_ITER $EXTRA_ARGS"
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

torchrun ${DISTRIBUTED_ARGS[@]} tools/run_text_generation_server.py   \
       --tensor-model-parallel-size $MP_SIZE  \
       --pipeline-model-parallel-size $PP_SIZE  \
       --expert-model-parallel-size $EP_SIZE \
       --load ${CHECKPOINT}  \
       --micro-batch-size ${BATCH_SIZE}  \
       --seed 42 \
       --bf16 \
       --port $SERVE_PORT \
       --flash-decode \
       --inference-max-seq-length 8192 \
       --return-log-probs \
       --use-checkpoint-args \
       $EXTRA_ARGS
