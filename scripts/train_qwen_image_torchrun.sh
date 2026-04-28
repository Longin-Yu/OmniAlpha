# Environment variables (override these to point to your local models/data)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Qwen/Qwen-Image-Edit-2509}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-/path/to/vae/checkpoint}"
# LORA_PATH="${LORA_PATH:-/path/to/lora/pytorch_lora_weights.safetensors}"

CYAN='\033[0;36m'
NC='\033[0m'

TIME_STR=$(date "+%Y-%m-%d_%H-%M-%S")

if [ -z "$MASTER_ADDR" ]; then
    printf "${CYAN}MASTER_ADDR not set, using localhost${NC}"
    MASTER_ADDR="localhost"
fi

if [ -z "$MASTER_PORT" ]; then
    printf "${CYAN}MASTER_PORT not set, using 29500${NC}"
    MASTER_PORT=29500
fi

printf "${CYAN}MASTER_PORT=$MASTER_PORT${NC}"

if [ -z "$NNODES" ]; then
    printf "${CYAN}NNODES not set, using 1${NC}"
    export NNODES=1
fi

if [ -z "$NPROC_PER_NODE" ]; then
    printf "${CYAN}NPROC_PER_NODE not set, using 8${NC}"
    export NPROC_PER_NODE=8
fi


if [ -z "$MACHINE_RANK" ]; then
    printf "${CYAN}MACHINE_RANK not set, using 0"
    export MACHINE_RANK=0
fi

if [ -z "$VERSION" ]; then
    printf "${CYAN}VERSION not set, please input: ${NC}"
    read VERSION
fi

printf "${CYAN}MACHINE_RANK=$MACHINE_RANK${NC}\n"
printf "${CYAN}MASTER_ADDR=$MASTER_ADDR${NC}\n"
printf "${CYAN}MASTER_PORT=$MASTER_PORT${NC}\n"
printf "${CYAN}NNODES=$NNODES${NC}\n"
printf "${CYAN}NPROC_PER_NODE=$NPROC_PER_NODE${NC}\n"
printf "${CYAN}VERSION=$VERSION${NC}\n"


export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="configs/deepspeed/zero1.json"
export ACCELERATE_DEEPSPEED_ZERO3_INIT=false


TORCHRUN_ARGS=(
    --nnodes=$NNODES
    --nproc_per_node=$NPROC_PER_NODE
    --node_rank=$MACHINE_RANK
    --master_addr=$MASTER_ADDR
    --master_port=$MASTER_PORT
)


# Model Configuration
MODEL_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --pretrained_vae_model "${VAE_MODEL_PATH}"
    --guidance_scale 1
    # --load_lora "${LORA_PATH}"
)

output_dir=runs/train/omni-alpha-lora-debug/v$VERSION

# Output Configuration
OUTPUT_ARGS=(
    --output_dir ${output_dir}/$TIME_STR
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --dataset_path "./configs/datasets.$VERSION.jsonc"
    --enable_weights False
)

# Training Configuration
TRAIN_ARGS=(
    # --gradient_checkpointing # memory efficient
    --rank 256 # lora rank follow ART
    --num_train_epochs 20 # number of training epochs
    --seed 42 # random seed
    --optimizer "AdamW"
    --learning_rate 5e-5
    --lr_scheduler "constant"
    --lr_warmup_steps 10

    #########   Please keep consistent with deepspeed config file ##########
    --train_batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
    --debug True # Enable debug mode for more frequent logging.
    --prob_drop_prompt 0.15
    --max_pixels 1048576
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 100
    --checkpointing_epochs 1
    --save_only_lora True
    # --checkpointing_limit 2 # maximum number of checkpoints to keep
    # --resume_from_checkpoint "/path/to/your/checkpoint"
)

# Validation Configuration
VALIDATION_ARGS=(
    --validation_steps 2500
    --eval_before_train True
    # --eval_show_reconstruction True
)

export CUDA_LAUNCH_BLOCKING=1

torchrun \
    "${TORCHRUN_ARGS[@]}" \
    tasks/diffusion/train_qwen_image_ti2i_lora.py \
        "${MODEL_ARGS[@]}" \
        "${OUTPUT_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${TRAIN_ARGS[@]}" \
        "${CHECKPOINT_ARGS[@]}" \
        "${VALIDATION_ARGS[@]}"
