#!/usr/bin/env bash
set -euo pipefail

# Multi-source preprocessing for RL latent dataset format.
# Output layout:
#   <OUTPUT_DIR>/prompt_embed/*.pt
#   <OUTPUT_DIR>/prompt_attention_mask/*.pt
#   <OUTPUT_DIR>/image_latents/*.pt
#   <OUTPUT_DIR>/preprocess.json

GPU_NUM=4
MASTER_PORT=19012

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL:-Qwen/Qwen-Image-Edit-2509}"
VAE_MODEL_PATH=/path/to/your/vae/hf
DATASET_CONFIG=tasks/rl/preprocess_mix_datasets.example.jsonc
SPLIT_NAME=train
OUTPUT_DIR=runs/preprocessed/rl_embeddings_rgba/mix_v1
NUM_WORKERS=4
MAX_PIXELS=1048576
SEED=42

torchrun --nproc_per_node="${GPU_NUM}" --master_port "${MASTER_PORT}" \
  tasks/rl/preprocess_rgba_MR_mix.py \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_PATH}" \
  --vae_model_path "${VAE_MODEL_PATH}" \
  --dataset_config "${DATASET_CONFIG}" \
  --split_name "${SPLIT_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_batch_size 1 \
  --dataloader_num_workers "${NUM_WORKERS}" \
  --max_pixels "${MAX_PIXELS}" \
  --seed "${SEED}"

