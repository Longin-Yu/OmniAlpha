#!/bin/bash
# Environment variables (override these to point to your local models/data)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen-Image-Edit-2509}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-/path/to/vae/checkpoint}"
LORA_PATH="${LORA_PATH:-/path/to/lora/pytorch_lora_weights.safetensors}"

CYAN='\033[0;36m'
NC='\033[0m'

TIME_STR=$(date "+%Y-%m-%d_%H-%M-%S")

if [ -z "$MASTER_ADDR" ]; then
    printf "${CYAN}MASTER_ADDR not set, using localhost${NC}\n"
    MASTER_ADDR="localhost"
fi

if [ -z "$MASTER_PORT" ]; then
    printf "${CYAN}MASTER_PORT not set, using 29500${NC}\n"
    MASTER_PORT=29500
fi

printf "${CYAN}MASTER_PORT=$MASTER_PORT${NC}\n"

if [ -z "$NNODES" ]; then
    printf "${CYAN}NNODES not set, using 1${NC}\n"
    export NNODES=1
fi

if [ -z "$NPROC_PER_NODE" ]; then
    printf "${CYAN}NPROC_PER_NODE not set, using 8${NC}\n"
    export NPROC_PER_NODE=8
fi


if [ -z "$MACHINE_RANK" ]; then
    printf "${CYAN}MACHINE_RANK not set, using 0${NC}\n"
    export MACHINE_RANK=0
fi

if [ -z "$DATA_VERSION" ]; then
    printf "${CYAN}DATA_VERSION not set, please input: ${NC}"
    read DATA_VERSION
fi

echo "================ Distributed Task Start ================"
printf "${CYAN}MACHINE_RANK=$MACHINE_RANK${NC}\n"
printf "${CYAN}MASTER_ADDR=$MASTER_ADDR${NC}\n"
printf "${CYAN}MASTER_PORT=$MASTER_PORT${NC}\n"
printf "${CYAN}NNODES=$NNODES${NC}\n"
printf "${CYAN}NPROC_PER_NODE=$NPROC_PER_NODE${NC}\n"
printf "${CYAN}DATA_VERSION=$DATA_VERSION${NC}\n"
printf "${CYAN}MODEL_VERSION=$MODEL_VERSION${NC}\n"
echo "========================================================"

TORCHRUN_ARGS=(
    --nnodes=$NNODES
    --nproc_per_node=$NPROC_PER_NODE
    --node_rank=$MACHINE_RANK
    --master_addr=$MASTER_ADDR
    --master_port=$MASTER_PORT
)

if ! declare -p EXTRA_PARAMS >/dev/null 2>&1; then
    EXTRA_PARAMS=""
fi

printf "${CYAN}EXTRA_PARAMS=$EXTRA_PARAMS${NC}\n"

# Example:
# EXTRA_PARAMS=" \
#     --no-use_r1_layer_blend_ssim \
#     --no-use_r1_layer_blend_psnr \
#     --no-use_r2_comp_ssim \
#     --no-use_r2_comp_psnr \
#     --no-use_r2_comp_lpips \
#     --no-use_r3_bg_blend_lpips \
#     --no-use_r4_fg_boundary_alpha_l1 \
# "


torchrun "${TORCHRUN_ARGS[@]}" \
    tasks/rl/grpo.py \
    --seed 42 \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
    --vae_model_path "${VAE_MODEL_PATH}" \
    --cache_dir runs/data/.cache \
    --data_json_path configs/datasets.$DATA_VERSION.jsonc \
    --dataset_split train \
    --max_pixels 1048576 \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 300 \
    --learning_rate 2e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 5 \
    --allow_tf32 \
    --cfg 0.0 \
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
    --selective_checkpointing 1.0 \
    --load_lora "${LORA_PATH}" \
    --save_only_lora \
    --vis_interval 5 \
    --vis_dir runs/train/grpo/$MODEL_VERSION/$TIME_STR/vis \
    --output_dir runs/train/grpo/$MODEL_VERSION/$TIME_STR/outputs \
    $EXTRA_PARAMS
