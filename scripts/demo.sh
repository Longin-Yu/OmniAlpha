#!/bin/bash
# Environment variables (override these to point to your local models)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen-Image-Edit-2509}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-/path/to/vae/checkpoint}"
LORA_PATH="${LORA_PATH:-/path/to/lora/pytorch_lora_weights.safetensors}"

python ./tasks/demo/gradio_demo.py \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --pretrained_vae_model "${VAE_MODEL_PATH}" \
  --lora_path "${LORA_PATH}" \
  --device cuda \
  --dtype bfloat16 \
  --output_dir ./outputs_gradio \
  --server_name 0.0.0.0 \
  --server_port 7860
