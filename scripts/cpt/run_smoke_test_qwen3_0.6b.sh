#!/bin/bash

# ============================================================================
# Qwen3-0.6B 发烟测试 (Smoke Test) 脚本
# 用于快速验证训练流程、数据处理、mask 和 position IDs 是否正确工作
# ============================================================================

set -e
export CUDA_VISIBLE_DEVICES=8
source /home/yulan_pretrain/gaoyanzipeng/pretrain-linear-moe/LLMBox/.venv/bin/activate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 添加项目目录到 PYTHONPATH，以便导入本地的 megatron.core
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ============================================================================
# CUDA 配置 - 使用 Context Parallel 或 Tensor Parallel 时必需
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ============================================================================
# 1. 环境配置
# ============================================================================

# 模型配置 - 使用 mcore 格式的模型
export MCORE_MODEL_CKPT="/media/public/models/huggingface/Qwen/Qwen3-0.6B-mcore"
export TOKENIZER_MODEL="/media/public/models/huggingface/Qwen/Qwen3-0.6B-mcore"

# 工作目录
export MLM_WORK_DIR="${MLM_WORK_DIR:-.}"
export MLM_MODEL_SAVE="${MLM_WORK_DIR}/checkpoints_smoke_test"

# 并行配置 (2 GPUs + Context Parallel)
export TP=${TP:-1}
export PP=${PP:-1}
export CP=${CP:-1}
export DP=${DP:-1}
export ETP=${ETP:-${TP}}
export EP=${EP:-1}

# 数据配置 - 小规模发烟测试
DATA_PATH="${DATA_PATH:-/mnt/yulan_pretrain/mount/data_final_train_qwen3/agent-clean/stage_1/yulan_input_ids_document}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-100}"
LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-100}"

# ============================================================================
# 2. 数据参数
# ============================================================================

MLM_DATA_ARGS=" \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-warmup-samples 0 \
    --train-data-path ${DATA_PATH} \
"

# ============================================================================
# 3. 训练参数 - 发烟测试配置
# ============================================================================

MLM_TRAIN_ARGS=" \
    --no-gradient-accumulation-fusion \
    --reset-position-ids \
    --reset-attention-mask \
    --eod-mask-loss \
    --micro-batch-size 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-check-for-nan-in-loss-and-grad \
    --document-packing-algorithm ffd \
"

# ============================================================================
# 4. 优化器参数
# ============================================================================

MLM_OPTIM_ARGS=" \
    --lr 2.0e-5 \
    --min-lr 1.0e-7 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std 0.010 \
"

# ============================================================================
# 5. 评估参数 - 发烟测试配置
# ============================================================================

MLM_EVAL_ARGS=" \
    --eval-iters 1 \
    --eval-interval 10 \
    --save-interval 100 \
    --log-interval 1 \
"

# ============================================================================
# 6. 模型参数 (Qwen3-0.6B)
# ============================================================================

MODEL_ARGS=" \
    --save-interval 100000 \
    --micro-batch-size 1 \
    --bf16 \
    --no-masked-softmax-fusion \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-rope-fusion \
    --normalization RMSNorm \
    --num-layers 28 \
    --hidden-size 1024 \
    --ffn-hidden-size 3072 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --seq-length 32768 \
    --max-position-embeddings 40960 \
    --tokenizer-type HuggingFaceTokenizer \
    --padded-vocab-size 151936 \
    --make-vocab-size-divisible-by 1 \
    --use-mcore-models \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --swiglu \
"

# ============================================================================
# 7. 启动训练
# ============================================================================

LAUNCH_SCRIPT="torchrun --nproc_per_node=$((ETP * EP * PP * CP * DP))"

PRETRAIN_SCRIPT="${PROJECT_ROOT}/pretrain_gpt.py"

echo "=========================================="
echo "启动 Qwen3-0.6B 发烟测试"
echo "=========================================="
echo "工作目录: ${MLM_WORK_DIR}"
echo "模型保存: ${MLM_MODEL_SAVE}"
echo "并行配置: TP=${TP} PP=${PP} CP=${CP} DP=${DP}"
echo "数据处理配置:"
echo "  - reset-position-ids: 启用"
echo "  - reset-attention-mask: 启用"
echo "  - eod-mask-loss: 启用"
echo "  - document-packing-algorithm: ffd"
echo "  - seq-length: 32768"
echo "  - context-parallel-size: 1"
echo "=========================================="
echo "提示: 设置 DEBUG_MASKS=1 环境变量可启用详细 debug 输出"
echo "  例如: DEBUG_MASKS=1 bash scripts/cpt/run_smoke_test_qwen3_0.6b.sh"
echo "=========================================="

${LAUNCH_SCRIPT} ${PRETRAIN_SCRIPT} \
    ${MODEL_ARGS} \
    --tensor-model-parallel-size ${TP} \
    --expert-tensor-parallel-size ${ETP} \
    --expert-model-parallel-size ${EP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${MCORE_MODEL_CKPT} \
    --finetune \
    --save ${MLM_MODEL_SAVE} \
    --ckpt-format torch_dist \
    --dist-ckpt-strictness log_unexpected \
    --no-load-optim \
    --no-load-rng \
    ${MLM_DATA_ARGS} \
    ${MLM_OPTIM_ARGS} \
    ${MLM_TRAIN_ARGS} \
    ${MLM_EVAL_ARGS} \
    --distributed-timeout-minutes 30 \
    --auto-detect-ckpt-format \
    --export-te-mcore-model

