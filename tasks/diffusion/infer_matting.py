import argparse
import copy
import itertools
import logging
import math
import os, json5 as json
import random
import shutil
import warnings
import functools
from contextlib import nullcontext
from pathlib import Path

from copy import deepcopy
from typing import *

import numpy as np
import torch
from torch import nn
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
# from diffusers import QwenImageEditPipeline

if is_wandb_available():
    import wandb

from alpha.args import ModelArguments, CustomTrainingArguments, DataArguments
from alpha.utils import alpha_blend, load_json_file
from alpha.vae.modeling import load_vae_from_local_dir
from alpha.pipelines.qwen_image_edit import CustomQwenImageEditPlusPipeline as QwenImageEditPlusPipeline
from alpha.pipelines.qwen_image_edit import QwenImageEditModules


model_path = "/path/to/Qwen-Image-Edit-2509"
vae_path = "/path/to/vae_rgba"
lora_path = "/path/to/lora_rgba/checkpoint.safetensors"

device = "cuda" 
dtype = torch.bfloat16
final_format = "RGBA" if vae_path else "RGB"

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    model_path,
) if not vae_path else QwenImageEditPlusPipeline.from_pretrained(
    model_path,
    vae=load_vae_from_local_dir(vae_path)
)

if lora_path:
    pipeline.load_lora_weights(lora_path)

pipeline = pipeline.to(device, dtype)

with torch.no_grad():
    prompt = "Pull out the foreground with fine edges and perfect transparency."
    image1 = Image.open("./inputs/matting_example_1.png").convert("RGBA")
    image2 = Image.open("./inputs/matting_example_2.png").convert("RGBA")
    
    outputs = pipeline(
        negative_prompt="",
        prompt=prompt,
        image=[image1, image2],
        num_inference_steps=50,
        height=512,
        width=512,
        true_cfg_scale=4,
        generator=torch.Generator(device).manual_seed(42),
        frames=1,
    ).images

    if outputs:
        save_path = "outputs/output_matting.png"
        outputs[0].save(save_path)
        print(f"图片已成功保存至: {os.path.abspath(save_path)}")
    else:
        print("生成失败，outputs 为空")