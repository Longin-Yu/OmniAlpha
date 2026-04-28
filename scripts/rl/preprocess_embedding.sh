#!/bin/bash
# Environment variables (override these to point to your local models/data)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen-Image-Edit-2509}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-/path/to/vae/checkpoint}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"

GPU_NUM=8 # 2,4,8
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL}"
DATA_PATH="${DATA_ROOT}/AIM-synthetic/AIM-synthetic_matting.jsonl"
IMAGE_DIR="${DATA_ROOT}/AIM-synthetic"
OUTPUT_DIR="runs/preprocessed/rl_embeddings_rgba/AIM-synthetic"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    tasks/rl/preprocess_rgba_MR.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
    --vae_model_path $VAE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --base_image_path $IMAGE_DIR \
    --prompt_dir $DATA_PATH
