from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, List
from transformers import HfArgumentParser
import os
import warnings

# -------------------------------
# 参数分组（按模型 / 数据 / 训练与运行）
# -------------------------------

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = None
    pretrained_vae_model: Optional[str] = field(
        default=None, metadata={"help": "Path to RGBA VAE model."}
    )
    revision: Optional[str] = field(
        default=None, metadata={"help": "HF hub revision of the pretrained model."}
    )
    variant: Optional[str] = field(
        default=None, metadata={"help": "Variant of model weights on hub, e.g., fp16."}
    )

    # 文本/生成相关
    max_sequence_length: int = field(
        default=512, metadata={"help": "Maximum sequence length for the T5 text encoder."}
    )
    guidance_scale: float = field(
        default=3.5, metadata={"help": "FLUX.1 dev is guidance-distilled; use ~3.5 by default."}
    )
    train_text_encoder: bool = field(
        default=False, metadata={"help": "Whether to train the text encoder (requires fp32)."}
    )

    # LoRA
    rank: int = field(
        default=4, metadata={"help": "LoRA rank (dimension of update matrices)."}
    )
    lora_layers: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Comma-separated transformer modules for LoRA, e.g. "to_k,to_q,to_v,to_out.0".'
        },
    )
    load_lora: Optional[str] = field(
        default=None, metadata={"help": "Path to LoRA weights to load (in HuggingFace format)."}
    )

    # 精度/保存
    mixed_precision: Optional[Literal["no", "fp16", "bf16"]] = field(
        default=None, metadata={"help": "Mixed precision to use (overrides accelerate config)."}
    )
    upcast_before_saving: bool = field(
        default=False,
        metadata={"help": "Upcast trained transformer layers to fp32 before saving."}
    )
    prior_generation_precision: Optional[Literal["no", "fp32", "fp16", "bf16"]] = field(
        default=None,
        metadata={"help": "Precision used for prior generation; defaults to fp16 on GPU else fp32."}
    )


@dataclass
class DataArguments:
    # 数据源（二选一，稍后会做互斥校验）
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to training datasets."}
    )

    # 先验保持（DreamBooth常见）
    with_prior_preservation: bool = field(
        default=False, metadata={"help": "Enable prior preservation loss."}
    )
    prior_loss_weight: float = field(
        default=1.0, metadata={"help": "Weight of prior preservation loss."}
    )
    class_data_dir: Optional[str] = field(
        default=None, metadata={"help": "Folder containing class images."}
    )
    class_prompt: Optional[str] = field(
        default=None, metadata={"help": "Prompt describing the class images."}
    )
    num_class_images: int = field(
        default=100, metadata={"help": "Minimum number of class images to maintain."}
    )

    # # 训练分辨率/增强
    # resolution: int = field(
    #     default=512,
    #     metadata={"help": "All images will be resized (then optionally cropped) to this resolution."},
    # )
    # center_crop: bool = field(
    #     default=False, metadata={"help": "Center-crop after resize; else random crop."}
    # )
    # random_flip: bool = field(
    #     default=False, metadata={"help": "Random horizontal flip."}
    # )

    # 采样加权
    weighting_scheme: Literal["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"] = field(
        default="none",
        metadata={"help": 'We default to "none" for uniform sampling and uniform loss.'},
    )
    logit_mean: float = field(default=0.0, metadata={"help": "Mean for 'logit_normal' scheme."})
    logit_std: float = field(default=1.0, metadata={"help": "Std for 'logit_normal' scheme."})
    mode_scale: float = field(
        default=1.29, metadata={"help": "Scale for 'mode' scheme (only effective for 'mode')."}
    )

    # 文本/验证
    # instance_prompt: str = field(
    #     default=None, metadata={"help": "Instance prompt, e.g. 'photo of a TOK dog', 'in the style of TOK'."}
    # )
    # validation_prompt: Optional[str] = field(
    #     default=None, metadata={"help": "Prompt used to generate validation images."}
    # )
    # num_validation_images: int = field(
    #     default=4, metadata={"help": "How many images to generate for validation."}
    # )
    validation_steps: int = field(
        default=50,
        metadata={
            "help": "Run validation every X epochs; generates num_validation_images each time."
        },
    )


@dataclass
class CustomTrainingArguments:
    debug: bool = field(
        default=False, metadata={"help": "Enable debug mode for more frequent logging."},
    )
    # 运行/输出/随机性
    output_dir: str = field(
        default="flux-dreambooth-lora", metadata={"help": "Where to write checkpoints & predictions."}
    )
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to cache downloaded models/datasets."}
    )

    # 批大小/轮次/步数
    train_batch_size: int = field(default=4, metadata={"help": "Per-device train batch size."})
    sample_batch_size: int = field(default=4, metadata={"help": "Per-device sampling batch size."})
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    max_train_steps: Optional[int] = field(
        default=None, metadata={"help": "Override epochs with a fixed number of steps."}
    )

    # Checkpoint
    checkpointing_steps: int = field(
        default=500, metadata={"help": "Save training state every X updates."}
    )
    checkpointing_epochs: int = field(
        default=5, metadata={"help": "Save a checkpoint every X epochs."}
    )
    checkpoints_total_limit: Optional[int] = field(
        default=None, metadata={"help": "Max number of checkpoints to keep."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Path saved by --checkpointing_steps, or "latest" to auto-pick last checkpoint.'
        },
    )
    save_only_lora: bool = field(
        default=False, metadata={"help": "Only save LoRA weights when saving checkpoints."}
    )

    # 优化器/学习率计划
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Accumulate steps before backward/update."}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Save memory at cost of slower backward."}
    )
    learning_rate: float = field(default=1e-4, metadata={"help": "Base learning rate."})
    text_encoder_lr: float = field(default=5e-6, metadata={"help": "LR for text encoder."})
    scale_lr: bool = field(
        default=False,
        metadata={"help": "Scale LR by #GPUs, grad_accum steps, and batch size."}
    )
    lr_scheduler: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = field(default="constant", metadata={"help": "LR scheduler type."})
    lr_warmup_steps: int = field(default=500, metadata={"help": "Warmup steps."})
    lr_num_cycles: int = field(default=1, metadata={"help": "Cycles for cosine_with_restarts."})
    lr_power: float = field(default=1.0, metadata={"help": "Power for polynomial scheduler."})

    # DataLoader
    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of subprocesses for data loading (0 = main process)."}
    )
    prob_drop_prompt: float = field(
        default=0.0, metadata={"help": "Probability of dropping text encoder input."}
    )

    # Optimizer 细节
    optimizer: Literal["AdamW", "prodigy"] = field(default="AdamW", metadata={"help": "Optimizer type."})
    use_8bit_adam: bool = field(
        default=False, metadata={"help": "Use bitsandbytes 8-bit Adam (AdamW only)."}
    )
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam/Prodigy."})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam/Prodigy."})
    prodigy_beta3: Optional[float] = field(
        default=None,
        metadata={"help": "Prodigy beta3; if None uses sqrt(beta2). Ignored if optimizer != 'prodigy'."},
    )
    prodigy_decouple: bool = field(
        default=True, metadata={"help": "Use decoupled weight decay (AdamW style)."}
    )
    adam_weight_decay: float = field(
        default=1e-4, metadata={"help": "Weight decay for UNet params."}
    )
    adam_weight_decay_text_encoder: float = field(
        default=1e-3, metadata={"help": "Weight decay for text encoder."}
    )
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Adam/Prodigy epsilon."})
    prodigy_use_bias_correction: bool = field(
        default=True, metadata={"help": "Use Adam bias correction (prodigy only)."}
    )
    prodigy_safeguard_warmup: bool = field(
        default=True, metadata={"help": "Safeguard D estimate during warmup (prodigy only)."}
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Gradient clipping norm."})

    # 记录/Hub/TF32
    hub_token: Optional[str] = field(default=None, metadata={"help": "HF hub token for pushing."})
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "Model repo id to sync with output_dir."}
    )
    logging_dir: str = field(default="logs", metadata={"help": "TensorBoard log dir."})
    report_to: str = field(
        default="tensorboard",
        metadata={
            "help": 'Where to report logs. One of "tensorboard", "wandb", "comet_ml", or "all".'
        },
    )
    allow_tf32: bool = field(
        default=False,
        metadata={"help": "Allow TF32 on Ampere GPUs for speedup."}
    )
    cache_latents: bool = field(
        default=False, metadata={"help": "Cache VAE latents to speed up training."}
    )

    # 分布式
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training."})


# -------------------------------
# 解析与校验
# -------------------------------

def validate_args(model_args: ModelArguments, data_args: DataArguments, train_args: CustomTrainingArguments):
    # dataset_name XOR instance_data_dir
    if (data_args.dataset_name is None) == (data_args.instance_data_dir is None):
        raise ValueError("Specify exactly one of `--dataset_name` or `--instance_data_dir`")

    # prior preservation 相关
    if data_args.with_prior_preservation:
        if data_args.class_data_dir is None:
            raise ValueError("You must specify --class_data_dir when --with_prior_preservation is set.")
        if data_args.class_prompt is None:
            raise ValueError("You must specify --class_prompt when --with_prior_preservation is set.")
    else:
        if data_args.class_data_dir is not None:
            warnings.warn("You do not need --class_data_dir without --with_prior_preservation.")
        if data_args.class_prompt is not None:
            warnings.warn("You do not need --class_prompt without --with_prior_preservation.")

    # 环境变量覆盖 local_rank（与原逻辑一致）
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != train_args.local_rank:
        train_args.local_rank = env_local_rank


def parse_args(input_args: Optional[List[str]] = None) -> Tuple[ModelArguments, DataArguments, CustomTrainingArguments]:
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    if input_args is None:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses(input_args)

    validate_args(model_args, data_args, train_args)
    return model_args, data_args, train_args
