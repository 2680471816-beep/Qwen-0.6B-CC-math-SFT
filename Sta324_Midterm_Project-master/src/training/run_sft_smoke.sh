#!/bin/bash
# SFT 训练启动脚本（小样本测试）
# 使用方法: ./run_sft_smoke.sh

set -e

# 激活 conda 环境
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate llm

# 设置环境变量
export LLM_RUNTIME_ROOT=/dev/shm/llm
export CUDA_VISIBLE_DEVICES=0

# WandB 配置（可选）
export WANDB_PROJECT=midterm_math_sft
export WANDB_RUN_NAME=smoke_test_100

# 训练参数
TRAIN_DATA="/home/ubuntu/Midterm_Project/outputs/coldstart_llm/coldstart_llm_A_sft_100.jsonl"
OUTPUT_DIR="/home/ubuntu/Midterm_Project/outputs/experiments/smoke_test_100"
MAX_EPOCHS=1
MAX_LEN=2048
LR=2e-5
BATCH_SIZE=2
GRAD_ACC=2

echo "=== 开始小样本 SFT 训练 ==="
echo "训练数据: $TRAIN_DATA"
echo "输出目录: $OUTPUT_DIR"
echo "Epochs: $MAX_EPOCHS, LR: $LR, MaxLen: $MAX_LEN"

# 运行训练
python /home/ubuntu/Midterm_Project/src/training/train_sft.py \
    --train-data "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --max-epochs $MAX_EPOCHS \
    --learning-rate $LR \
    --max-len $MAX_LEN \
    --micro-batch-size $BATCH_SIZE \
    --gradient-accumulation $GRAD_ACC \
    --train-gpu 0

echo "=== 训练完成 ==="