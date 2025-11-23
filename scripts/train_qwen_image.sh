TIME_STR=$(date "+%Y-%m-%d_%H-%M-%S")
MASTER_PORT=$((RANDOM % (30000 - 20000 + 1) + 20000))

echo MASTER_PORT=$MASTER_PORT

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# Model Configuration
MODEL_ARGS=(
    --pretrained_model_name_or_path /path/to/Qwen/Qwen-Image-Edit-2509
    --pretrained_vae_model /path/to/vae/checkpoint
    --guidance_scale 1
    --load_lora /path/to/lora/checkpoint/pytorch_lora_weights.safetensors
)

# TODO: fix load lora
output_dir=/path/to/omni-alpha-lora

# Output Configuration
OUTPUT_ARGS=(
    --output_dir ${output_dir}/$TIME_STR
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --dataset_path "./configs/datasets.jsonc"
)

# Training Configuration
TRAIN_ARGS=(
    # --gradient_checkpointing # memory efficient
    --rank 256 # lora rank follow ART
    --num_train_epochs 20000 # number of training epochs
    --seed 42 # random seed
    --optimizer "AdamW"
    # --learning_rate 2e-5
    --learning_rate 5e-5
    --lr_scheduler "constant"
    --lr_warmup_steps 10

    #########   Please keep consistent with deepspeed config file ##########
    --train_batch_size 1
    --gradient_accumulation_steps 1 # flux diffusers choice
    # --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
    --debug True # Enable debug mode for more frequent logging.
    --prob_drop_prompt 0.15
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 2500 # save checkpoint every x steps
    --checkpointing_epochs 1000
    --save_only_lora True
    # --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/path/to/your/checkpoint"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --validation_steps 2500
)

export CUDA_LAUNCH_BLOCKING=1

accelerate launch --config_file=configs/accelerate.yaml --main_process_port=$MASTER_PORT tasks/diffusion/train_qwen_image_ti2i_lora.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"