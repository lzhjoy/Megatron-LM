#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Dir Arguments
DIR=`pwd`
PRETRAINED_CKPT_ROOT_PATH=${PRETRAINED_CKPT_ROOT_PATH:-"/volume/ailab4sci/txie/huyiwen/megatron_lm_workspace"}
PRETRAINED_CKPT_ID=${PRETRAINED_CKPT_ID:-"NOT_EXISTS"}
PRETRAINED_CKPT_NAME=${PRETRAINED_CKPT_NAME:-"NOT_EXISTS"}
OUTPUT_CHECKPOINT_PATH=${OUTPUT_CHECKPOINT_PATH:-"/volume/ailab4sci/txie/huyiwen/megatron_lm_workspace"}
OUTPUT_BASE_PATH=${OUTPUT_BASE_PATH:-"/volume/ailab4sci/txie/huyiwen/megatron_lm_workspace"}

# Training Arguments
SEQ_LEN=8192
BATCH_SIZE=${BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4096}
MP_SIZE=${MP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-4}
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-"false"}
LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

# Learning Rate Arguments
LR=${LR:-"2e-3"}
MIN_LR=${MIN_LR:-"3.0e-5"}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-"linear"}
TRAIN_TOKENS=${TRAIN_TOKENS:-1_000_000_000}
LR_WARMUP_TOKENS=${LR_WARMUP_TOKENS:-10_000_000}
LR_DECAY_TOKENS=${LR_DECAY_TOKENS:-990_000_000}

# Sample-based training
TRAIN_SAMPLES=$(( ${TRAIN_TOKENS//_/}  / ${SEQ_LEN} ))
LR_DECAY_SAMPLES=$(( ${LR_DECAY_TOKENS//_/}  / ${SEQ_LEN} ))
LR_WARMUP_SAMPLES=$(( ${LR_WARMUP_TOKENS//_/}  / ${SEQ_LEN} ))

# MoE Arguments
MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-768}
MOE_TOPK=${MOE_TOPK:-2}
NUM_EXPERTS=${NUM_EXPERTS:-16}
NUM_SHARED_EXPERTS=${NUM_SHARED_EXPERTS:-0}
LOAD_BALANCING=${LOAD_BALANCING:-"aux_loss"}
MOE_EXPERT_CAPACITY_FACTOR=${MOE_EXPERT_CAPACITY_FACTOR:-0.0}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.05}

# Model Arguments
INIT_STD=${INIT_STD:-0.006}
NUM_LAYERS=${NUM_LAYERS:-12}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
NUM_ATTN_HEADS=16
NUM_QUERY_GROUPS=2
ROTARY_BASE=${ROTARY_BASE:-"100000"}
TIE_EMBEDDING=${TIE_EMBEDDING:-"true"}

# Multi-node Arguments
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${PET_NNODES:-"1"}
NODE_RANK=${PET_NODE_RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

EXTRA_ARGS=${EXTRA_ARGS:-""}

# ###################################################
# ################# Process Arguments
# ###################################################

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
NAME="mixtral-0.5b-${MODEL_SIZE}-q${NUM_ATTN_HEADS}-kv${NUM_QUERY_GROUPS}-ep-${NUM_EXPERTS}-sep-${num_shared_experts}-top${MOE_TOPK}-cf-${MOE_EXPERT_CAPACITY_FACTOR}-mlc-${MOE_AUX_LOSS_COEFF}-bf16-ep${EP_SIZE}-mp${MP_SIZE}-pp${PP_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${GPU_SIZE}-seqlen-${SEQ_LEN}"
CHECKPOINT_PATH="${OUTPUT_CHECKPOINT_PATH}/checkpoint/${NAME}"
TENSORBOARD_DIR="${OUTPUT_CHECKPOINT_PATH}/tensorboard/${NAME}_${current_time}"
LOG_DIR="${OUTPUT_CHECKPOINT_PATH}/log/${NAME}_${current_time}"
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${LOG_DIR}
cp $0 ${LOG_DIR}

# check continual-pretrain or from-scratch
if [ -d "$CHECKPOINT_PATH" ]; then
    LOAD_CHECKPOINT_PATH="${CHECKPOINT_PATH}"
    CONTINUE_TRAIN=${CONTINUE_TRAIN:-'true'}
    echo "Find existing checkpoint $CHECKPOINT_PATH"
else
    LOAD_CHECKPOINT_PATH="${PRETRAINED_CKPT_ROOT_PATH}/${PRETRAINED_CKPT_NAME}"
    CONTINUE_TRAIN=${CONTINUE_TRAIN:-'false'}
    echo "Checkpoint $CHECKPOINT_PATH does not exists. Try to load from $LOAD_CHECKPOINT_PATH"
fi

# setup tokenizer
TOKENIZER_TYPE=${TOKENIZER_TYPE:-'hf_tokenizer_qwen'}
DATA_PATH_CACHE="/volume/ailab4sci/txie/huyiwen/cache"
if [[ ${TOKENIZER_TYPE} == "hf_tokenizer_qwen" ]]; then
    DATA_PATH_TOKENIZED="${DATA_PATH}/qwen2.5"
    TOKENIZER_ARGS="--tokenizer-type HuggingFaceTokenizer --tokenizer_model ../../tokenizer"
elif [[ ${TOKENIZER_TYPE} == "gpt2bpe" ]]; then
    DATA_PATH_TOKENIZED="${DATA_PATH}"
    TOKENIZER_ARGS="--vocab-file /volume/ailab4sci/models/gpt2/vocab.json --merge-file /volume/ailab4sci/models/gpt2/merges.txt"
else
    echo "ERROR: Unknown tokenizer type ${TOKENIZER_TYPE}"
    exit 1
fi

# setup embedding tying
if [[ "1${TIE_EMBEDDING}" == "1false" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} \
        --untie-embeddings-and-output-weights
    "
fi

# ###################################################
# ################# models
# ###################################################


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)


MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTN_HEADS}
    --init-method-std ${INIT_STD}
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --group-query-attention
    --num-query-groups ${NUM_QUERY_GROUPS}
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base ${ROTARY_BASE}
    --use-flash-attn
)

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --expert-tensor-parallel-size 1
    --moe-grouped-gemm
    --moe-router-topk ${MOE_TOPK}
    --moe-router-load-balancing-type ${LOAD_BALANCING}
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-expert-capacity-factor ${MOE_EXPERT_CAPACITY_FACTOR}
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF}
)

TRAINING_ARGS=(
    --micro-batch-size ${BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr ${LR}
    --train-samples ${TRAIN_SAMPLES}
    --lr-warmup-samples ${LR_WARMUP_SAMPLES}
    --lr-decay-samples ${LR_DECAY_SAMPLES}
    --lr-decay-style ${LR_DECAY_STYLE}
    --min-lr ${MIN_LR}
    --split 100,0,0
    --weight-decay 0.1
    --clip-grad 0.5
    --num-workers 2
    --bf16
    --save ${CHECKPOINT_PATH}
    --load ${LOAD_CHECKPOINT_PATH}
)

DATA_ARGS=(
    --data-path ${DATA_PATH_TOKENIZED}
    --data-cache-path ${DATA_PATH_CACHE}
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${MP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --expert-model-parallel-size ${EP_SIZE}
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL}
    --log-throughput
    --save-interval ${SAVE_INTERVAL}
    --eval-interval 1000
    --eval-iters 10
    --tensorboard-dir ${TENSORBOARD_DIR}
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"DSV3"}
        --wandb-exp-name ${NAME}
    )
fi

if [ "1${ACTIVATION_CHECKPOINT}" = "1true" ]; then
EXTRA_ARGS="${EXTRA_ARGS} \
        --recompute-granularity selective
        "
fi

set -x

torchrun ${DISTRIBUTED_ARGS[@]} ../../pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS} \
    ${EXTRA_ARGS} 2>&1 | tee ${LOG_DIR}/LOG_NODE_RANK_${NODE_RANK}.log
