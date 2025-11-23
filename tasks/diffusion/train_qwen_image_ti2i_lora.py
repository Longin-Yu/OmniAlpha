#!/usr/bin/env python
"""
Modified from 
- https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py
- https://github.com/FlyMyAI/flymyai-lora-trainer
"""

# import alpha.inplace

import argparse
import copy
import itertools
import logging
import math
import os, json
import random, pickle
import shutil
import warnings
import functools
from contextlib import nullcontext
from pathlib import Path

from copy import deepcopy
from typing import *

import PIL.Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler

import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

# from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # QwenImageEditPipeline,
    QwenImageTransformer2DModel,
    AutoencoderKLQwenImage,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser
from transformers.image_utils import ChannelDimension

from accelerate.utils import gather_object

if is_wandb_available():
    import wandb

from alpha.args import ModelArguments, CustomTrainingArguments, DataArguments
from alpha.utils import alpha_blend, load_json_file
from alpha.vae.modeling import load_vae_from_local_dir
from alpha.pipelines.qwen_image_edit import CustomQwenImageEditPlusPipeline as QwenImageEditPlusPipeline
from alpha.pipelines.qwen_image_edit import QwenImageEditModules
from alpha.data import DatasetManager

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

# logger = logging.getLogger(__name__)
logger = get_logger(__name__, log_level="INFO")

@dataclass
class TrainingArguments(ModelArguments, CustomTrainingArguments, DataArguments):
    @staticmethod
    def validate(args: "TrainingArguments"):
        if args.pretrained_model_name_or_path is None:
            raise ValueError("You must specify a pretrained model name or path.")

        if args.train_text_encoder:
            raise NotImplementedError("Training text encoder is not supported yet.")
        
        if args.cache_latents:
            raise NotImplementedError("Caching latents is not supported yet.")
        
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != args.local_rank:
            args.local_rank = env_local_rank
            
        if args.train_batch_size != 1:
            raise NotImplementedError("Only batch size of 1 is supported for now. (because of multi resolutions)")

        if args.with_prior_preservation:
            raise NotImplementedError("Prior preservation is not supported yet.")
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
        else:
            # logger is not available yet
            if args.class_data_dir is not None:
                warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
            if args.class_prompt is not None:
                warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

def convert_to_rgb(img: PIL.Image.Image) -> PIL.Image.Image:
    """
    将任何 PIL 图像（包括 RGBA）转换为 'RGB' 模式，
    使用白色背景进行混合。
    """
    if img.mode == 'RGB':
        return img
    
    if img.mode == 'RGBA':
        # 创建一个白色背景
        bg = PIL.Image.new("RGB", img.size, (255, 255, 255))
        # 使用 RGBA 图像的 alpha 通道作为蒙版进行粘贴
        bg.paste(img, (0, 0), img)
        return bg
    else:
        # 转换其他模式（如 'L', 'P' 等）
        return img.convert('RGB')

def create_image_grid(data_list: List[Dict], gap: int = 10) -> Optional[PIL.Image.Image]:
    """
    根据您的规范，从数据列表创建复杂的网格图像。
    - 验证所有输入图像必须是 RGBA 模式。
    - 每一行 (item) 包含: [input_images..., output_images..., predictions...]
    - 自动处理可变的行长（不再使用占位符填充）。
    - 自动计算每列的最大宽度和每行的最大高度。
    - 在白色和黑色背景上创建并排的双重网格。
    """
    
    all_rows_images: List[List[PIL.Image.Image]] = []
    all_rows_locations: List[List[Tuple[int, int]]] = []
    max_width = 0
    current_y = 0

    # --- 1. 收集所有图像并验证 RGBA ---
    for item in data_list:
        row_images: List[PIL.Image.Image] = []
        row_locations: List[Tuple[int, int]] = []
        current_x = 0
        row_max_height = 0
        for key in ["input_images", "output_images", "predictions"]:
            if key in item:
                for img in item[key]:
                    assert isinstance(img, PIL.Image.Image)
                    if img.mode != 'RGBA':
                        raise ValueError(f"图像 {key}.{img} 必须是 RGBA 模式，但检测到: {img.mode}。")
                    row_images.append(img)
                    row_locations.append((current_x, current_y))
                    current_x += img.width + gap
                    row_max_height = max(row_max_height, img.height)
            current_x += gap * 3  # 每个类别之间的额外间隙
        max_width = max(max_width, current_x)
        current_y += row_max_height + gap
        all_rows_images.append(row_images)
        all_rows_locations.append(row_locations)

    # --- 4. 创建最终的画布 (Canvas) ---
    # 尺寸计算逻辑与之前相同
    
    # 创建双倍宽度的画布，左白右黑
    grid_image = PIL.Image.new('RGB', (max_width * 2 + gap, current_y), (255, 255, 255))
    black_bg = PIL.Image.new('RGB', (max_width, current_y), (0, 0, 0)) # 注意：黑色背景宽度应为 total_width
    grid_image.paste(black_bg, (max_width + gap, 0)) # 粘贴时考虑 gap


    # --- 5. 将所有图像粘贴到网格上 (不再有占位符) ---
    # for i in range(len(all_rows_images)): # 遍历每一行
        
    #     # 我们必须遍历 max_cols 以保持列对齐
    #     for j in range(max_cols): 
            
    #         # 关键：检查该单元格 (i, j) 是否真的有图像
    #         if j < len(all_rows_images[i]):
    #             img = all_rows_images[i][j]
                
    #             # 粘贴到左侧 (白色背景)
    #             # 使用 img 作为 mask (蒙版)，因为我们已验证它是 RGBA
    #             grid_image.paste(img, (current_x_left, current_y), img)
                
    #             # 粘贴到右侧 (黑色背景)
    #             grid_image.paste(img, (current_x_right, current_y), img)
    
    for row_images, row_locations in zip(all_rows_images, all_rows_locations):
        for img, (x, y) in zip(row_images, row_locations):
            # 粘贴到左侧 (白色背景)
            grid_image.paste(img, (x, y), img)
            # 粘贴到右侧 (黑色背景)
            grid_image.paste(img, (x + max_width + gap, y), img)

    return grid_image

def log_validation(
    pipeline: QwenImageEditPlusPipeline,
    accelerator: Accelerator,
    val_dataloader: Dataset,
    global_step: int,
    seed: int,
):
    with accelerator.autocast():
        device = accelerator.device
        data = []
        for batch in val_dataloader:
            prompts = batch["prompts"]
            input_images = batch.get("input_images", None)
            if input_images:
                input_images = input_images[0]
            frames = len(batch["output_images"][0])
            height = batch["output_images"][0][0].height
            width = batch["output_images"][0][0].width
            assert len(batch["prompts"]) == 1, "Batch size greater than 1 is not supported for validation."
            
            predictions = pipeline(
                negative_prompt="",
                prompt=prompts,
                image=input_images,
                num_inference_steps=50,
                height=height,
                width=width,
                true_cfg_scale=4,
                # debug=True,
                generator=torch.Generator(device).manual_seed(seed),
                frames=frames,
            ).images
            
            data.append({
                "prompt": prompts[0],
                "input_images": batch.get("input_images", [[]])[0],
                "output_images": batch["output_images"][0],
                "predictions": predictions,
            })
    
    data_from_all_processes = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(data_from_all_processes, data)
    data = []
    for proc_data in data_from_all_processes:
        # data.extend(pickle.loads(proc_data))
        data.extend(proc_data)
    
    """
    Compose the following structure into a big figure:
        List of {
            "prompt": str,
            "input_images": List[PIL.Image],
            "outputs_images": List[PIL.Image],
            "predictions": List[PIL.Image],
        }
    """
    
    # print(data)
    
    grid_image_pil = create_image_grid(data, gap=8)
    
    text_to_log = ""
    prompt_lines = []
    for i, item in enumerate(data):
        prompt_lines.append(f"**Item {i}:** {item.get('prompt', '')}")
    text_to_log = "\n\n".join(prompt_lines)

    for tracker in accelerator.trackers:
        
        # 我们只对 TensorBoard tracker 感兴趣
        if tracker.name == "tensorboard":
            writer = tracker.writer # 获取底层的 SummaryWriter

            # A. 记录网格图像
            if grid_image_pil is not None:
                try:
                    # 将 PIL (H, W, C) 转换为 Numpy (C, H, W)
                    grid_image_np = np.array(grid_image_pil).transpose(2, 0, 1)
                    
                    # 添加到 TensorBoard
                    writer.add_image(
                        "Validation/Image_Grid", # Tagn
                        grid_image_np,
                        global_step=global_step
                    )
                except Exception as e:
                    accelerator.print(f"[ERROR] 无法记录图像到 TensorBoard: {e}")

            # B. 记录 Prompts 文本
            if text_to_log:
                try:
                    writer.add_text(
                        "Validation/Prompts", # Tag
                        text_to_log,
                        global_step=global_step
                    )
                except Exception as e:
                    accelerator.print(f"[ERROR] 无法记录文本到 TensorBoard: {e}")

class TI2IDataset(Dataset):
    
    exist_paths: Set[str] = set()
    
    def __init__(
        self,
        image_dir: str,
        data_path: Optional[str] = None,
        data: Optional[List[Dict]] = None,
    ):
        if data is None and data_path is None:
            raise ValueError("You must specify either `data` or `data_path`.")
        if data is not None and data_path is not None:
            raise ValueError("You must specify only one of `data` or `data_path`.")
        
        self.data = data or []
        self.data_path = data_path
        self.image_dir = image_dir
        if data_path is not None:
            with open(data_path, 'r') as df:
                if data_path.endswith('.jsonl'):
                    data_items = [json.loads(line) for line in df]
                else:
                    data_items = json.load(df)
            self.inplace_and_verify_data(data_items, image_dir)
            self.data.extend(data_items)
        
    def inplace_and_verify_data(self, data_items, image_dir):
        """
        Verify:
        1. all items have this schema: {"prompt": str, "input_images": List[str], "output_images": List[str]}
        2. all images exist
        """
        for item in data_items:
            if "prompt" not in item or "input_images" not in item or "output_images" not in item:
                raise ValueError("Each data item must have 'prompt', 'input_images', and 'output_images' fields.")
            if not isinstance(item["prompt"], str):
                raise ValueError("'prompt' must be a string.")
            if not isinstance(item["input_images"], list) or not all(isinstance(i, str) for i in item["input_images"]):
                raise ValueError("'input_images' must be a list of strings.")
            if not isinstance(item["output_images"], list) or not all(isinstance(i, str) for i in item["output_images"]):
                raise ValueError("'output_images' must be a list of strings.")
            for lst in [item["input_images"], item["output_images"]]:
                for idx in range(len(lst)):
                    img_path = lst[idx]
                    full_path = os.path.join(image_dir, img_path)
                    if full_path not in TI2IDataset.exist_paths and not os.path.exists(full_path):
                        raise FileNotFoundError(f"Image file {full_path} does not exist.")
                    TI2IDataset.exist_paths.add(full_path)
                    lst[idx] = full_path
    
    # def preprocess_image(self, image):
    #     image = exif_transpose(image)
    #     if not image.mode == "RGBA":
    #         image = image.convert("RGBA")
    #     return transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(image))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        example = {}
        input_images = []
        for img_path in item["input_images"]:
            image = Image.open(img_path).convert("RGBA")
            # image = self.preprocess_image(image)
            input_images.append(image)
        if len(input_images) > 0:
            # input_images = torch.stack(input_images)
            example["input_images"] = input_images  # (N, C, H, W)

        output_images = []
        for img_path in item["output_images"]:
            image = Image.open(img_path).convert("RGBA")
            # image = self.preprocess_image(image)
            output_images.append(image)
        # output_images = torch.stack(output_images)
        example["output_images"] = output_images  # (M, C, H, W)

        example["prompt"] = item["prompt"]
        return example

def collate_fn(examples):
    assert len(examples) == 1, "Batch size greater than 1 is not supported."
    prompts = [example["prompt"] for example in examples]
    output_images = [example["output_images"] for example in examples]
    ret = {
        "prompts": prompts,
        "output_images": output_images,
    }
    if "input_images" in examples[0]:
        # ret["input_images"] = torch.stack([example["input_images"] for example in examples])
        ret["input_images"] = [example["input_images"] for example in examples]
    return ret


def log_tensor(tensor: torch.Tensor, name: str):
    print("============")
    print(f"[TENSOR: {name=}] {tensor.shape=}, {tensor.dtype=}, {tensor.device=}, {tensor.max()=}, {tensor.min()=}, {tensor.mean()=}, {tensor.std()=}")
    print("============")

def main(args: TrainingArguments):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    # logger.info(accelerator.state)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
    )
    
    logger.info("Loaded pretrained model from {}".format(args.pretrained_model_name_or_path))
    
    modules = QwenImageEditModules.from_pipeline(pipeline)
    if args.pretrained_vae_model is not None:
        modules.vae = load_vae_from_local_dir(args.pretrained_vae_model)
    
    # processor_cls = type(modules.processor.image_processor)
    # processor_cls.__call__ = functools.partial(
    #     processor_cls.__call__, input_data_format=ChannelDimension.FIRST
    # )
    # modules.processor.image_processor 
    
    pipeline = QwenImageEditPlusPipeline(**modules.to_dict())
    
    # TODO: remove after debug
    for param in pipeline.transformer.parameters():
        assert param.requires_grad == True

    # We only train the additional adapter LoRA layers
    modules.requires_grad_(False)

    # TODO: remove after debug
    for param in pipeline.transformer.parameters():
        assert param.requires_grad == False

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
        
    modules.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        modules.transformer.enable_gradient_checkpointing()

    if args.load_lora:
        pipeline.load_lora_weights(args.load_lora, "default")
        # modules.transformer.load_lora_adapter(args.load_lora)
    else:
        if args.lora_layers is not None:
            target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
        else:
            target_modules = [
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ]
        # now we will add new LoRA weights the transformer layers
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        modules.transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                model = unwrap_model(model)
                if isinstance(model, type(unwrap_model(modules.transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            QwenImageEditPlusPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(modules.transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = QwenImageEditPlusPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [modules.transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, modules.transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    dataset_manager = DatasetManager(args.dataset_path).set_default_class(TI2IDataset)
    train_dataset, train_weights = dataset_manager.get_split("train", enable_weight=True, verbose=True)
    valid_dataset = dataset_manager.get_split("valid", verbose=True)
    
    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of validation examples: {len(valid_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        # shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
        sampler=WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    modules.transformer, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        modules.transformer, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux-dev-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    raw_scheduler = modules.scheduler

    def get_sigmas(timesteps: torch.Tensor, n_dim=4, dtype=torch.float32):
        sigmas = modules.scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = modules.scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # vae_scale_factor = 2 ** len(modules.vae.temperal_downsample)

    for epoch in range(first_epoch, args.num_train_epochs):
            
        modules.transformer.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(modules.transformer):
                with accelerator.autocast():
                    modules.scheduler = deepcopy(raw_scheduler)
                    with torch.no_grad():
                        
                        assert len(batch["prompts"]) == 1, "Batch size greater than 1 is not supported."
                        
                        # Step 1. Prepare Prompts
                        prompts = []
                        for prompt in batch["prompts"]:
                            if random.random() < args.prob_drop_prompt:
                                prompts.append("")
                            else:
                                prompts.append(prompt)
                        
                        # Step 2. Prepare Images and Latents
                        output_latents = [] # (B, C, F, H, W)
                        for batch_outputs in batch["output_images"]:
                            vae_image, vae_image_size = pipeline.prepare_images(
                                batch_outputs, "vae", reshape=False
                            ) # vae_image: List of (B=1, C, F=1, H, W) on F dim
                            vae_latents = [pipeline._encode_vae_image(img.to(accelerator.device, weight_dtype), generator=None) for img in vae_image]
                            vae_latents = torch.concat(vae_latents, dim=2)
                            output_latents.append(vae_latents)
                        output_latents = torch.concat(output_latents, dim=0)
                        
                        has_input_images = "input_images" in batch
                        if has_input_images:
                            input_images = batch["input_images"][0]
                            input_prompt_images, input_image_shapes = pipeline.prepare_images(
                                input_images, "condition", reshape=False
                            )
                            input_vae_images, input_vae_shapes = pipeline.prepare_images(
                                input_images, "vae", reshape=False
                            ) # List of (B=1, C, F=1, H, W)
                            input_latents = [pipeline._encode_vae_image(img.to(accelerator.device, weight_dtype), generator=None) for img in input_vae_images]
                            input_latents = torch.concat(input_latents, dim=2)
                            
                            # input_images = batch["input_images"].to(dtype=weight_dtype).to(accelerator.device)  # (B, F, C, H, W)
                            # blended_input_images = alpha_blend(input_images, 1., alpha_min=-1).to(dtype=weight_dtype).to(accelerator.device) # (B, C, F, H, W)
                            # input_images = input_images.permute(0, 2, 1, 3, 4) # (B, C, F, H, W)
                            # input_latents = modules.vae.encode(input_images).latent_dist.sample()
                            
                        # Step 3. Prepare Model Input

                        bsz = output_latents.shape[0]
                        noise = torch.randn_like(output_latents, device=accelerator.device, dtype=weight_dtype)
                        u = compute_density_for_timestep_sampling(
                            weighting_scheme="none",
                            batch_size=bsz,
                            logit_mean=0.0,
                            logit_std=1.0,
                            mode_scale=args.mode_scale,
                        )
                        indices = (u * modules.scheduler.config.num_train_timesteps).long()
                        # print(f"{indices=}")
                        # print(f"{modules.scheduler.timesteps.shape=}")
                        # print(f"{modules.scheduler.timesteps[indices]=}")
                        timesteps = modules.scheduler.timesteps[indices].to(device=output_latents.device)

                        sigmas = get_sigmas(timesteps, n_dim=output_latents.ndim, dtype=output_latents.dtype)
                        # print(f"{sigmas.shape=}")
                        noisy_output_latents = (1.0 - sigmas) * output_latents + sigmas * noise
                        # print(f"{noisy_model_input.shape=}")
                        # Concatenate across channels.
                        # pack the latents.
                        if has_input_images:
                            packed_input_latents = QwenImageEditPlusPipeline.pack_latents_multi_frames(input_latents) # (B, L, C)
                        packed_noisy_output_latents = QwenImageEditPlusPipeline.pack_latents_multi_frames(noisy_output_latents) # (B, L, C)
                        
                        img_shapes = [(
                            1,
                            noisy_output_latents.shape[-2] // 2,
                            noisy_output_latents.shape[-1] // 2
                        )] * noisy_output_latents.shape[-3]
                        
                        # print(f"{output_latents.shape=}")
                        # print(f"{noisy_output_latents.shape=}")
                        # print(f"{noise.shape=}")
                        # print(f"{img_shapes=}")
                        
                        if has_input_images:
                            img_shapes += [
                                *tuple((
                                    1, 
                                    input_latents.shape[-2] // 2,
                                    input_latents.shape[-1] // 2
                                ) for _t in range(input_latents.shape[-3])),
                            ]
                            
                        img_shapes = [img_shapes] * bsz
                        
                        # print(f"POST {img_shapes=}")
                        # print(f"POST {packed_noisy_output_latents.shape=}")
                        
                        packed_noisy_model_input_concated = (
                            torch.cat([packed_noisy_output_latents, packed_input_latents], dim=1)
                            if has_input_images else packed_noisy_output_latents
                        )
                        
                        # print(f"{packed_noisy_model_input_concated.shape=}")
                        
                        # print(f"{blended_input_images.shape=}")
                        
                        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                            prompt=prompts,
                            image=input_prompt_images if has_input_images else None, # concat on width
                            device=accelerator.device,
                        )
                        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                        
                    # print(f"{packed_noisy_model_input_concated.shape=}")
                    # print(f"{timesteps.shape=}")
                    # print(f"{prompt_embeds_mask.shape=}")
                    # print(f"{prompt_embeds.shape=}")
                    # print(f"{img_shapes=}")
                    # print(f"{txt_seq_lens=}")

                    model_pred = modules.transformer(
                        hidden_states=packed_noisy_model_input_concated,
                        timestep=timesteps / 1000,
                        guidance=None,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        return_dict=False,
                    )[0]
                    model_pred = model_pred[:, :packed_noisy_output_latents.size(1)]
                    # print(f"{model_pred.shape=}")
                    model_pred = QwenImageEditPlusPipeline.unpack_latents_multi_frames(
                        model_pred,
                        *output_latents.shape[-3:],
                    )
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                    # flow-matching loss
                    target = noise - output_latents
                    loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
                    loss = loss.mean()
                    # Gather the losses across all processes for logging (if we use distributed training).
                    # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    # train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(modules.transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if not args.save_only_lora:
                            accelerator.save_state(save_path)
                        else:
                            if accelerator.is_main_process:
                                t = unwrap_model(modules.transformer)
                                if args.upcast_before_saving:
                                    t.to(torch.float32)
                                else:
                                    t = t.to(weight_dtype)
                                transformer_lora_layers = get_peft_model_state_dict(t)

                                QwenImageEditPlusPipeline.save_lora_weights(
                                    save_directory=save_path,
                                    transformer_lora_layers=transformer_lora_layers,
                                )
                        logger.info(f"Saved state to {save_path}")
                        
                if global_step % args.validation_steps == 0:
                    
                    # print(batch)

                    accelerator.wait_for_everyone()
                    val_modules = modules.to_dict()
                    val_modules["transformer"] = unwrap_model(modules.transformer)
                    val_modules["scheduler"] = deepcopy(raw_scheduler)
                    val_pipeline = QwenImageEditPlusPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                        **val_modules,
                    ).to(accelerator.device, dtype=weight_dtype)
                    
                    with torch.no_grad():
                        log_validation(val_pipeline, accelerator, valid_dataloader, global_step, seed=args.seed)
                    
                    del val_pipeline, val_modules
                    free_memory()
                        
            accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break
        
        if accelerator.sync_gradients:
            if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                if epoch > 0 and epoch % args.checkpointing_epochs == 0:
                    # TODO remove transformer
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    t = unwrap_model(modules.transformer)
                    if args.upcast_before_saving:
                        t.to(torch.float32)
                    else:
                        t = t.to(weight_dtype)
                    transformer_lora_layers = get_peft_model_state_dict(t)

                    save_path = os.path.join(args.output_dir, f"epochs-{epoch}")
                    QwenImageEditPlusPipeline.save_lora_weights(
                        save_directory=save_path,
                        transformer_lora_layers=transformer_lora_layers,
                    )

    # Save the lora layers
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     transformer = unwrap_model(transformer)
    #     if args.upcast_before_saving:
    #         transformer.to(torch.float32)
    #     else:
    #         transformer = transformer.to(weight_dtype)
    #     transformer_lora_layers = get_peft_model_state_dict(transformer)

    #     if args.train_text_encoder:
    #         text_encoder_one = unwrap_model(text_encoder_one)
    #         text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
    #     else:
    #         text_encoder_lora_layers = None

    #     QwenImageEditPipeline.save_lora_weights(
    #         save_directory=args.output_dir,
    #         transformer_lora_layers=transformer_lora_layers,
    #         text_encoder_lora_layers=text_encoder_lora_layers,
    #     )

    #     # Final inference
    #     # Load previous pipeline
    #     pipeline = QwenImageEditPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         revision=args.revision,
    #         variant=args.variant,
    #         torch_dtype=weight_dtype,
    #         vae=vae,
    #     )
    #     # load attention processors
    #     pipeline.load_lora_weights(args.output_dir)

    #     # run inference
    #     images = []
    #     if args.validation_prompt and args.num_validation_images > 0:
    #         pipeline_args = {"prompt": args.validation_prompt}
    #         images = log_validation(
    #             pipeline=pipeline,
    #             args=args,
    #             accelerator=accelerator,
    #             pipeline_args=pipeline_args,
    #             global_step=global_step,
    #             is_final_validation=True,
    #             torch_dtype=weight_dtype,
    #         )

    #     images = None
    #     del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args: TrainingArguments = HfArgumentParser(TrainingArguments).parse_args()
    print(args)
    TrainingArguments.validate(args)
    main(args)
    # args = parse_args()
    # main(args)