#!/bin/bash
set -e

PYTHON_BIN="/path/to/miniconda3/envs/OmniAlpha/bin/python"
SCRIPT_PATH="./tasks/diffusion/infer.py"

MODEL_PATH="${PRETRAINED_MODEL:-Qwen/Qwen-Image-Edit-2509}"
VAE_PATH="/path/to/models/OmniAlpha/rgba_vae"
LORA_PATH="/path/to/models/OmniAlpha/rgba_lora/pytorch_lora_weights.safetensors"

TASK_ID="test"
OUTPUT_DIR="/path/to/outputs/${TASK_ID}"
SEED="42"

PROMPT="Generate a transparent-background RGBA image. A white cat with sunglasses, full body, distant view."
NEG_PROMPT=""

mkdir -p "${OUTPUT_DIR}"

INPUT_IMAGES=(
  # "/path/to/image1.png"
  # "/path/to/image2.jpg"
)

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --pretrained_model_name_or_path "${MODEL_PATH}"
  --pretrained_vae_model "${VAE_PATH}"
  --lora_path "${LORA_PATH}"
  --prompts "${PROMPT}"
  --negative_prompt "${NEG_PROMPT}"
  --output_dir "${OUTPUT_DIR}"
  --task_id "${TASK_ID}"
  --seed "${SEED}"
  --frames 1
  --num_images_per_prompt 1
  --height 1024
  --width 1024
  --num_inference_steps 50
  --true_cfg_scale 4.0
  --device cuda
  --dtype bfloat16
)

if [ ${#INPUT_IMAGES[@]} -gt 0 ]; then
  CMD+=(--input_images "${INPUT_IMAGES[@]}")
fi

printf 'Running command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"