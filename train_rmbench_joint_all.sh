#!/bin/bash
# One-click SimVLA fine-tuning on all RMBench demo_clean tasks.
#
# Default split:
# - first 40 episodes per task -> train
# - next 10 episodes per task  -> eval meta only
#
# This script is intended for server-side training where the environment matches local.

set -euo pipefail

GPU_IDS=${1:-0}
OUTPUT_DIR=${2:-./runs/simvla_rmbench_joint_all}
INIT_CKPT=${3:-../SimCkpt}
SMOLVLM_MODEL=${4:-HuggingFaceTB/SmolVLM-500M-Instruct}
BATCH_SIZE=${5:-8}
TASK_CONFIG=${6:-demo_clean}
TRAIN_EPISODES_PER_TASK=${7:-40}
EVAL_EPISODES_PER_TASK=${8:-10}

DATA_DIR=${DATA_DIR:-../RMBench/data/data}
TRAIN_META_PATH=${TRAIN_META_PATH:-./datasets/metas/rmbench_all_train_${TRAIN_EPISODES_PER_TASK}of$((TRAIN_EPISODES_PER_TASK+EVAL_EPISODES_PER_TASK)).json}
EVAL_META_PATH=${EVAL_META_PATH:-./datasets/metas/rmbench_all_eval_${EVAL_EPISODES_PER_TASK}of$((TRAIN_EPISODES_PER_TASK+EVAL_EPISODES_PER_TASK)).json}
NORM_STATS_PATH=${NORM_STATS_PATH:-./norm_stats/rmbench_all_train_${TRAIN_EPISODES_PER_TASK}of$((TRAIN_EPISODES_PER_TASK+EVAL_EPISODES_PER_TASK))_joint_norm.json}

LEARNING_RATE=${LEARNING_RATE:-1e-4}
LEARNING_COEF=${LEARNING_COEF:-0.1}
NUM_ACTIONS=${NUM_ACTIONS:-10}
ITERS=${ITERS:-100000}
WARMUP_STEPS=${WARMUP_STEPS:-0}
FREEZE_STEPS=${FREEZE_STEPS:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
DEPTH=${DEPTH:-24}
NUM_HEADS=${NUM_HEADS:-16}
IMAGE_SIZE=${IMAGE_SIZE:-384}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29514}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
WANDB_PROJECT=${WANDB_PROJECT:-simvla-rmbench}
WANDB_API_KEY=${WANDB_API_KEY:-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}

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_PROJECT
export WANDB_API_KEY

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_PROCESSES=${#GPU_ARRAY[@]}

echo "============================================================"
echo "SimVLA RMBench Joint Fine-tuning"
echo "============================================================"
echo "GPU_IDS: ${GPU_IDS}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "INIT_CKPT: ${INIT_CKPT}"
echo "SMOLVLM_MODEL: ${SMOLVLM_MODEL}"
echo "DATA_DIR: ${DATA_DIR}"
echo "TASK_CONFIG: ${TASK_CONFIG}"
echo "TRAIN_EPISODES_PER_TASK: ${TRAIN_EPISODES_PER_TASK}"
echo "EVAL_EPISODES_PER_TASK: ${EVAL_EPISODES_PER_TASK}"
echo "TRAIN_META_PATH: ${TRAIN_META_PATH}"
echo "EVAL_META_PATH: ${EVAL_META_PATH}"
echo "NORM_STATS_PATH: ${NORM_STATS_PATH}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_API_KEY: [set]"
echo "============================================================"

mkdir -p "$(dirname "${TRAIN_META_PATH}")"
mkdir -p "$(dirname "${EVAL_META_PATH}")"
mkdir -p "$(dirname "${NORM_STATS_PATH}")"
mkdir -p "${OUTPUT_DIR}"

echo "[1/3] Creating train meta..."
python create_rmbench_meta.py \
    --data_dir "${DATA_DIR}" \
    --task_config "${TASK_CONFIG}" \
    --split train \
    --train_episodes_per_task "${TRAIN_EPISODES_PER_TASK}" \
    --eval_episodes_per_task "${EVAL_EPISODES_PER_TASK}" \
    --output "${TRAIN_META_PATH}"

echo "[2/3] Creating eval meta..."
python create_rmbench_meta.py \
    --data_dir "${DATA_DIR}" \
    --task_config "${TASK_CONFIG}" \
    --split eval \
    --train_episodes_per_task "${TRAIN_EPISODES_PER_TASK}" \
    --eval_episodes_per_task "${EVAL_EPISODES_PER_TASK}" \
    --output "${EVAL_META_PATH}"

echo "[3/3] Computing train-split normalization statistics..."
python compute_rmbench_norm_stats.py \
    --data_dir "${DATA_DIR}" \
    --task_config "${TASK_CONFIG}" \
    --split train \
    --train_episodes_per_task "${TRAIN_EPISODES_PER_TASK}" \
    --eval_episodes_per_task "${EVAL_EPISODES_PER_TASK}" \
    --output "${NORM_STATS_PATH}"

ARGS="--models ${INIT_CKPT} \
    --output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_META_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode rmbench_joint \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size ${IMAGE_SIZE} \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm ${MAX_GRAD_NORM}"

echo "============================================================"
echo "Launching training..."
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=${HF_HOME:-$(pwd)/.hf} \
HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$(pwd)/.hf/hub} \
accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    train_smolvlm.py ${ARGS}

echo "Training completed."
echo "Train meta: ${TRAIN_META_PATH}"
echo "Eval meta: ${EVAL_META_PATH}"
echo "Norm stats: ${NORM_STATS_PATH}"
