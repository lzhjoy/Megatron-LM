#!/bin/bash

# ============================================================================
# Qwen3-0.6B HuggingFace 格式转换为 Megatron 格式
# ============================================================================
# 
# 用途：将 HuggingFace 格式的 Qwen3-0.6B 模型转换为 Megatron 格式
# 
# 使用方法：
#   bash scripts/cpt/convert_qwen3_0.6b_hf_to_megatron.sh \
#     --hf-model-path /path/to/hf/model \
#     --output-path /path/to/megatron/model \
#     --tp 1
#
# ============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 激活虚拟环境（如果存在）
VENV_PATH="/home/yulan_pretrain/gaoyanzipeng/pretrain-linear-moe/LLMBox/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
fi
export PYTHONPATH="/home/yulan_pretrain/lvzhihao/YuLan-Pretrain:${PYTHONPATH}"


# ============================================================================
# 参数解析
# ============================================================================

HF_MODEL_PATH="/media/public/models/huggingface/Qwen/Qwen3-0.6B"
OUTPUT_PATH="/media/public/models/huggingface/Qwen/Qwen3-0.6B-mcore"
TP=1
PP=1
BF16=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-model-path)
            HF_MODEL_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --pp)
            PP="$2"
            shift 2
            ;;
        --fp16)
            BF16=false
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$HF_MODEL_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "错误：缺少必需参数"
    echo "用法: $0 --hf-model-path <path> --output-path <path> [--tp <num>] [--pp <num>]"
    exit 1
fi

# ============================================================================
# 转换配置
# ============================================================================

echo "=========================================="
echo "Qwen3-0.6B HuggingFace 到 Megatron 格式转换"
echo "=========================================="
echo "HF 模型路径: ${HF_MODEL_PATH}"
echo "输出路径: ${OUTPUT_PATH}"
echo "Tensor Parallel: ${TP}"
echo "Pipeline Parallel: ${PP}"
echo "BF16: ${BF16}"
echo "=========================================="

# 创建输出目录
mkdir -p "${OUTPUT_PATH}"

# ============================================================================
# 执行转换
# ============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1

cd "${PROJECT_ROOT}"

python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader qwen \
    --saver core \
    --load-dir "${HF_MODEL_PATH}" \
    --save-dir "${OUTPUT_PATH}" \
    --tokenizer-model "${HF_MODEL_PATH}" \
    --checkpoint-type hf \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size ${PP} \
    $([ "$BF16" = "true" ] && echo "--bf16" || echo "--fp16")

echo "=========================================="
echo "✅ 转换完成！"
echo "Megatron 格式模型已保存到: ${OUTPUT_PATH}"
echo "=========================================="

