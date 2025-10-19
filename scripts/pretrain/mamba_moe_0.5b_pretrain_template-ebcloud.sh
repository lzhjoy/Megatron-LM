#!/bin/bash
# Mamba 0.5B total MoE
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="./triton-cache/mamba-moe/"

# Dir Arguments
DIR=`pwd`
PRETRAINED_CKPT_ROOT_PATH=${PRETRAINED_CKPT_ROOT_PATH:-"/data/pretrain-linear-moe/megatron_lm_workspace"}
PRETRAINED_CKPT_ID=${PRETRAINED_CKPT_ID:-"NOT_EXISTS"}
PRETRAINED_CKPT_NAME=${PRETRAINED_CKPT_NAME:-"NOT_EXISTS"}
OUTPUT_CHECKPOINT_PATH=${OUTPUT_CHECKPOINT_PATH:-"/data/pretrain-linear-moe/megatron_lm_workspace"}

# Training Arguments
SEQ_LEN=${SEQ_LEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4096}
MP_SIZE=${MP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-"false"}
LOG_INTERVAL=${LOG_INTERVAL:-1}

# Learning Rate Arguments
LR=${LR:-"2e-3"}
MIN_LR=${MIN_LR:-"3.0e-5"}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-"linear"}
TRAIN_TOKENS=${TRAIN_TOKENS:-1_000_000_000}
LR_WARMUP_TOKENS=${LR_WARMUP_TOKENS:-10_000_000}
LR_DECAY_TOKENS=${LR_DECAY_TOKENS:-990_000_000}
SAVE_TOKENS=${SAVE_TOKENS:-1_000_000_000}

# Sample-based training
TRAIN_SAMPLES=$(( ${TRAIN_TOKENS//_/}  / ${SEQ_LEN} ))
LR_DECAY_SAMPLES=$(( ${LR_DECAY_TOKENS//_/}  / ${SEQ_LEN} ))
LR_WARMUP_SAMPLES=$(( ${LR_WARMUP_TOKENS//_/}  / ${SEQ_LEN} ))
SAVE_INTERVAL=$(( ${SAVE_TOKENS//_/} / ${SEQ_LEN} / ${GLOBAL_BATCH_SIZE} ))

# MoE Arguments
MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-768}
MOE_TOPK=${MOE_TOPK:-2}
NUM_EXPERTS=${NUM_EXPERTS:-16}
NUM_SHARED_EXPERTS=${NUM_SHARED_EXPERTS:-0}
LOAD_BALANCING=${LOAD_BALANCING:-"dsv3"}
MOE_ROUTER_SCORE_FUNCTION=${MOE_ROUTER_SCORE_FUNCTION:-"sigmoid"}
MOE_EXPERT_CAPACITY_FACTOR=${MOE_EXPERT_CAPACITY_FACTOR:-2}
MOE_ROUTER_BIAS_UPDATE_RATE=${MOE_ROUTER_BIAS_UPDATE_RATE:-1e-3}

# Model Arguments
INIT_STD=${INIT_STD:-0.02}
NUM_LAYERS=${NUM_LAYERS:-12}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-16}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-2}
ROTARY_BASE=${ROTARY_BASE:-"100000"}
TIE_EMBEDDING=${TIE_EMBEDDING:-"true"}

# Multi-node Arguments
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${PET_NNODES:-"1"}
NODE_RANK=${PET_NODE_RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# hybrid mamba arguments
MAMBA_HEAD_DIM=${MAMBA_HEAD_DIM:-64}
MAMBA_NUM_GROUPS=${MAMBA_NUM_GROUPS:-6}
MAMBA_STATE_DIM=${MAMBA_STATE_DIM:-128}
MAMBA_EXPAND=${MAMBA_EXPAND:-2}
FREEZE_NON_MAMBA=${FREEZE_NON_MAMBA:-false}
HYBRID_OVERRIDE_PATTERN_TYPE=${HYBRID_OVERRIDE_PATTERN_TYPE:-None}
HYBRID_MLP_RATIO=${HYBRID_MLP_RATIO:-0.5}

EXTRA_ARGS=${EXTRA_ARGS:-""}

# ###################################################
# ################# Process Arguments
# ###################################################

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
JOB_ID=${TASK_UUID:-$current_time}
MODEL_SIZE=${MODEL_SIZE:-"unknown_size"}
NAME="${NAME_PREFIX}mamba_hybrid-${MODEL_SIZE}-${NUM_LAYERS}layers-q${NUM_ATTN_HEADS}-kv${NUM_QUERY_GROUPS}-hybrid${HYBRID_ATTN}-pattern_${HYBRID_OVERRIDE_PATTERN_TYPE}-mheaddim${MAMBA_HEAD_DIM}-mnumgroups${MAMBA_NUM_GROUPS}-mstatedim${MAMBA_STATE_DIM}-mexpand${MAMBA_EXPAND}-freeze_${FREEZE_NON_MAMBA}-ep${EP_SIZE}-mp${MP_SIZE}-pp${PP_SIZE}-cp${CP_SIZE}-lr${LR}-minlr${MIN_LR}-bs${GLOBAL_BATCH_SIZE}-gpus${WORLD_SIZE}-seqlen${SEQ_LEN}"
CHECKPOINT_PATH="${OUTPUT_CHECKPOINT_PATH}/checkpoint/${NAME}"
LOG_DIR="${OUTPUT_CHECKPOINT_PATH}/log/${JOB_ID}_${NAME}"
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${LOG_DIR}
ln -s $CHECKPOINT_PATH $LOG_DIR/checkpoint
echo $JOB_ID >> $CHECKPOINT_PATH/linked_runs.txt
cp $LAUNCH_SCRIPT_PATH ${LOG_DIR}
cp $0 ${LOG_DIR}
mkdir -p ${CHECKPOINT_PATH}/launch_script
cp $LAUNCH_SCRIPT_PATH ${CHECKPOINT_PATH}/launch_script
cp $0 ${CHECKPOINT_PATH}/launch_script

# ottn_only / attn_mamba
if [ -n "${LOAD_FROM_CHECKPOINT}" ] && ( [ "${LOAD_FROM_CHECKPOINT}" = "attn_only" ] || [ "${LOAD_FROM_CHECKPOINT}" = "attn_mamba" ] ); then
    # assert "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" does not exist
    if [ -z "${CHECKPOINT_LOAD_PATH}" ]; then
        echo "ERROR: CHECKPOINT_LOAD_PATH is not set"
        exit 1
    fi
    if [ -f "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" ]; then
        echo -e "\033[31mCheckpoint '$CHECKPOINT_PATH' exists. Please check if you want to continue training from Yulan-Mini checkpoint.\033[0m"
        exit 1
    fi
    LOAD_CHECKPOINT_PATH="${CHECKPOINT_LOAD_PATH}"
    echo -e "\033[32mLoad from Yulan-Mini checkpoint $CHECKPOINT_LOAD_PATH\033[0m"
elif [ -f "$CHECKPOINT_PATH/latest_checkpointed_iteration.txt" ]; then
    LOAD_CHECKPOINT_PATH="${CHECKPOINT_PATH}"
    CONTINUE_TRAIN=${CONTINUE_TRAIN:-'true'}
    echo -e "\033[32mFind existing checkpoint $CHECKPOINT_PATH\033[0m"
else
    LOAD_CHECKPOINT_PATH="${PRETRAINED_CKPT_ROOT_PATH}/${PRETRAINED_CKPT_NAME}"
    CONTINUE_TRAIN=${CONTINUE_TRAIN:-'false'}
    echo -e "\033[32mCheckpoint '$CHECKPOINT_PATH' does not exists. Try to load from '$LOAD_CHECKPOINT_PATH'\033[0m"
fi

# setup tokenizer
TOKENIZER_TYPE=${TOKENIZER_TYPE:-'hf_tokenizer_qwen'}
DATA_PATH_CACHE="/data/pretrain-linear-moe-dev/cache/datasets/data_cache"
if [[ ${TOKENIZER_TYPE} == "hf_tokenizer_qwen" ]]; then
    DATA_PATH_TOKENIZED=${DATA_PATH_TOKENIZED:-"${DATA_PATH}/qwen2.5"}
    TOKENIZER_ARGS="--tokenizer-type HuggingFaceTokenizer --tokenizer-model ../../tokenizer"
elif [[ ${TOKENIZER_TYPE} == "gpt2bpe" ]]; then
    DATA_PATH_TOKENIZED=${DATA_PATH_TOKENIZED:-"${DATA_PATH}"}
    TOKENIZER_ARGS="--vocab-file /volume/ailab4sci/models/gpt2/vocab.json --merge-file /volume/ailab4sci/models/gpt2/merges.txt"
elif [[ ${TOKENIZER_TYPE} == "hf_tokenizer_yulan_mini" ]]; then
    DATA_PATH_TOKENIZED=${DATA_PATH_TOKENIZED:-"${DATA_PATH}/yulan_mini"}
    TOKENIZER_ARGS="--tokenizer-type HuggingFaceTokenizer --tokenizer-model /data/pretrain-linear-moe/preprocess/modify_tokenizer/YuLan-Mini-Qwen-Template"
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

# # moe
# if [[ ${LOAD_BALANCING} == "dsv3" ]]; then
#     EXTRA_ARGS="${EXTRA_ARGS} \
#         --moe-router-enable-expert-bias
#     "
#     LOAD_BALANCING=none
# fi
# if [ -n "$MOE_AUX_LOSS_COEFF" ]; then
#     echo "ERROR: DeepSeek V3 does not support MOE_AUX_LOSS_COEFF=$MOE_AUX_LOSS_COEFF"
#     exit 1
# fi

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
    --hybrid-attention-ratio ${HYBRID_ATTN}
    --hybrid-mlp-ratio ${HYBRID_MLP_RATIO}
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_moe_stack_spec
    --disable-bias-linear
    --add-qkv-bias
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
    --mamba-head-dim ${MAMBA_HEAD_DIM}
    --mamba-num-groups ${MAMBA_NUM_GROUPS}
    --mamba-state-dim ${MAMBA_STATE_DIM}
    --mamba-expand ${MAMBA_EXPAND}
)

if [ -n "${HYBRID_OVERRIDE_PATTERN_TYPE}" ]; then
    if [ "${HYBRID_OVERRIDE_PATTERN_TYPE}" = "M0" ]; then
        HYBRID_OVERRIDE_PATTERN="M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-"
        MODEL_ARGS+=(
            --hybrid-override-pattern ${HYBRID_OVERRIDE_PATTERN}
        )
    elif [ "${HYBRID_OVERRIDE_PATTERN_TYPE}" = "A0" ]; then
        HYBRID_OVERRIDE_PATTERN="*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-" # 112
        # HYBRID_OVERRIDE_PATTERN="*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-" # 96
        # HYBRID_OVERRIDE_PATTERN="*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-" # 80
        # HYBRID_OVERRIDE_PATTERN="*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-" # 64
        MODEL_ARGS+=(
            --hybrid-override-pattern ${HYBRID_OVERRIDE_PATTERN}
        )
    elif [ "${HYBRID_OVERRIDE_PATTERN_TYPE}" = "A01" ]; then
        HYBRID_OVERRIDE_PATTERN="*-*-M-M-M-M-M-M-M-M-M-M-M-M-M-M-*-*-M-M-M-M-M-M-M-M-M-M-M-M-M-M-*-*-M-M-M-M-M-M-M-M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-"
        MODEL_ARGS+=(
            --hybrid-override-pattern ${HYBRID_OVERRIDE_PATTERN}
        )
    elif [ "${HYBRID_OVERRIDE_PATTERN_TYPE}" = "M01" ]; then
        HYBRID_OVERRIDE_PATTERN="M-M-M-M-M-M-M-*-*-M-M-M-M-M-M-M-M-M-M-M-M-M-M-*-*-M-M-M-M-M-M-M-M-M-M-M-M-M-M-*-*-M-M-M-M-M-M-M-M-M-M-M-M-M-M-*-"
        MODEL_ARGS+=(
            --hybrid-override-pattern ${HYBRID_OVERRIDE_PATTERN}
        )
    elif [ "${HYBRID_OVERRIDE_PATTERN_TYPE}" = "Nemo_A7_M49_F49" ]; then
        HYBRID_OVERRIDE_PATTERN="M-M-M-M-M-M*-M-M-M-M-M-M*-M-M-M-M-M-M*-M-M-M-M-M-M*-M-M-M-M-M-M*-M-M-M-M-M-M*-M-M-M-M-M-M*-M-M-M-M-M-M-M-"
        MODEL_ARGS+=(
            --hybrid-override-pattern ${HYBRID_OVERRIDE_PATTERN}
        )
    elif [ "${HYBRID_OVERRIDE_PATTERN_TYPE}" = "yulanmini" ]; then
        HYBRID_OVERRIDE_PATTERN="*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
        MODEL_ARGS+=(
            --hybrid-override-pattern ${HYBRID_OVERRIDE_PATTERN}
        )
    fi
    echo -e "\033[32mHYBRID_OVERRIDE_PATTERN: ${HYBRID_OVERRIDE_PATTERN}\033[0m"
fi

# MOE_ARGS=(
#     --num-experts ${NUM_EXPERTS}
#     --expert-tensor-parallel-size 1
#     --moe-grouped-gemm
#     --moe-router-topk ${MOE_TOPK}
#     --moe-router-load-balancing-type ${LOAD_BALANCING}
#     --moe-router-score-function sigmoid
#     --moe-token-dispatcher-type alltoall
#     --overlap-param-gather
#     --overlap-grad-reduce
#     --moe-expert-capacity-factor ${MOE_EXPERT_CAPACITY_FACTOR}
#     --moe-router-bias-update-rate ${MOE_ROUTER_BIAS_UPDATE_RATE}
# )

MOE_ARGS=(
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
    --overlap-param-gather
    --overlap-grad-reduce
)

if [ "1${FREEZE_NON_MAMBA}" = "1true" ]; then
    TRAINING_ARGS+=(
        --freeze-non-mamba
    )
fi

DATA_ARGS=(
    --data-path ${DATA_PATH_TOKENIZED}
    --data-cache-path ${DATA_PATH_CACHE}
    --no-create-attention-mask-in-dataloader
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${MP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --expert-model-parallel-size ${EP_SIZE}
    --use-distributed-optimizer
    --sequence-parallel
    --context-parallel-size ${CP_SIZE}
)

LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL}
    --log-throughput
    --save-interval ${SAVE_INTERVAL}
    --eval-interval 1000
    --eval-iters 10
    --tensorboard-dir ${LOG_DIR}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
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

if [ $NODE_RANK == "0" ]; then
    which torchrun >> ${LOG_DIR}/ENV-${HOSTNAME}.log
    python -V >> ${LOG_DIR}/ENV-${HOSTNAME}.log
    pip list >> ${LOG_DIR}/ENV-${HOSTNAME}.log
    env >> ${LOG_DIR}/ENV-${HOSTNAME}.log
    echo $(which torchrun) ${DISTRIBUTED_ARGS[@]} ../../pretrain_mamba.py ${MODEL_ARGS[@]} ${DATA_ARGS[@]} ${MOE_ARGS[@]} ${TRAINING_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${LOGGING_ARGS[@]} ${TOKENIZER_ARGS} ${EXTRA_ARGS} >> ${LOG_DIR}/ENV-${HOSTNAME}.log
fi
set -x

torchrun ${DISTRIBUTED_ARGS[@]} ../../pretrain_mamba.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS} \
    ${EXTRA_ARGS} 2>&1 | tee ${LOG_DIR}/LOG_NODE_RANK_${NODE_RANK}.log

if [ $NODE_RANK == "0" ]; then
    cp ${LOG_DIR}/LOG_NODE_RANK_${NODE_RANK}.log ${CHECKPOINT_PATH}/LOG_NODE_RANK_${NODE_RANK}.log
fi