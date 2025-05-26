# ------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER=whoami
source /volume/ailab4sci/miniconda/bin/activate
conda activate megatron
# ------------------

set -eo pipefail
# ------------------

cd /volume/ailab4sci/txie/huyiwen/Ubiquant-Pretrain/scripts/pretrain

export SUFFIX="test_17_yulan_mini"  # 数据集名称，一般格式为 v1.1.1_xxx_xxx_xxx

OUTPUT_CHECKPOINT_PATH="/volume/ailab4sci/txie/huyiwen/megatron_lm_workspace" \
DATA_PATH="/volume/ailab4sci/txie/huyiwen/dataset/${SUFFIX}" \
BATCH_SIZE=16 GLOBAL_BATCH_SIZE=1024 \
TRAIN_TOKENS=40_000_000_000 LR_WARMUP_TOKENS=1_000_000_000 SAVE_TOKENS=10_000_000_000 \
LR_DECAY_STYLE='linear' LR_DECAY_TOKENS=40_000_000_000 \
LR=2e-3 MIN_LR=7e-7 \
MP_SIZE=2 PP_SIZE=1 \
TOKENIZER_TYPE="hf_tokenizer_yulan_mini" \
ACTIVATION_CHECKPOINT='true' \
EXTRA_ARGS="" \
bash dsv3_0.5b_pretrain_template.sh