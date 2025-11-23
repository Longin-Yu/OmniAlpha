TIME_STR=$(date "+%Y-%m-%d_%H-%M-%S")
MASTER_PORT=$((RANDOM % (30000 - 20000 + 1) + 20000))

echo MASTER_PORT=$MASTER_PORT
export NCCL_IB_DISABLE=1

# export WANDB_MODE=offline
setting=vae_main
VAE_dir="/path/to/vae/${TIME_STR}-${setting}"
train_data_dir=(
    "/path/to/your/dataset1"
    "/path/to/your/dataset2"
)
pretrained_vae_path="/path/to//Qwen/Qwen-Image-Edit-2509/vae_rgba"

accelerate launch --config_file=configs/accelerate.yaml \
    --num_processes=8 --main_process_port=$MASTER_PORT tasks/vae/finetune.py \
    --pretrained_path ${pretrained_vae_path} \
    --train_data_dir "${train_data_dir[@]}" \
    --num_eval 8 \
    --output_dir ${VAE_dir} \
    --train_batch_size 2 \
    --num_train_epochs 100 \
    --gan_start_step 4000 \
    --learning_rate 1.5e-5 \
    --resolution 1024 \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --checkpointing_steps 2500 \
    --validation_steps 500 \
    --mixed_precision bf16 \
    --enable_frame true \
    --report_to tensorboard \
    --config configs/experiments/vae/${setting}.yaml