#!/bin/bash
# Environment variables (override these to point to your local models/data)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen-Image-Edit-2509}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-/path/to/vae/checkpoint}"
DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
LORA_PATH="${LORA_PATH:-/path/to/lora/pytorch_lora_weights.safetensors}"

torchrun --nproc_per_node=4 --master_port 19002 -m \
    tasks.rl.grpo \
    --seed 42 \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
    --vae_model_path "${VAE_MODEL_PATH}" \
    --cache_dir data/.cache \
    --data_json_path "${DATA_ROOT}/rl_embeddings/preprocess.json" \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 300 \
    --learning_rate 2e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 100 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/train_grpo_outputs \
    --num_output_frames 2 \
    --reward_gt_jsonl "${DATA_ROOT}/PrismLayers_5imgs/PrismLayers_layerdecompose_train_2layers.jsonl" \
    --reward_gt_root "${DATA_ROOT}/PrismLayers_5imgs" \
    --sampling_steps 20 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 12627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --w_r1_layer_blend_ssim 1.0 \
    --w_r1_layer_blend_psnr 1.0 \
    --w_r2_comp_ssim 1.0 \
    --w_r2_comp_psnr 1.0 \
    --w_r2_comp_lpips 1.0 \
    --w_r3_bg_blend_lpips 1.0 \
    --w_r4_fg_boundary_alpha_l1 1.0 \
    --boundary_alpha_low 0.05 \
    --boundary_alpha_high 0.95 \
    --boundary_blur_px 10 \
    --psnr_max_db 50.0 \
    --num_generations 8 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --selective_checkpointing 1.0\
    --load_lora "${LORA_PATH}" \
    --save_only_lora \
    --vis_interval 1 \
    --vis_dir data/train_grpo_outputs/train_vis
