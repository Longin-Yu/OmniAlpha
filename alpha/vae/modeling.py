import argparse, os, json
from typing import *

import torch.nn as nn
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage, QwenImageCausalConv3d, QwenImageEncoder3d, QwenImageCausalConv3d
from diffusers.configuration_utils import ConfigMixin, register_to_config

available_classes = [
    AutoencoderKL,
    AutoencoderKLQwenImage
]
AutoencoderClass = Union[AutoencoderKL, AutoencoderKLQwenImage, "AutoencoderKLQwenImageAlpha"]

def register_to_available_classes(cls):
    available_classes.append(cls)
    return cls

def load_vae_from_local_dir(model_path: str, config_rel_path: str = "config.json"):
    config_path = os.path.join(model_path, config_rel_path)
    with open(config_path, "r") as f:
        config = json.load(f)
    assert "_class_name" in config, f"config.json must contain _class_name field, but got {config.keys()}"
    class_name = config["_class_name"]
    cls = next((c for c in available_classes if c.__name__ == class_name), None)
    if cls is None:
        raise ValueError(f"class {class_name} not found in available classes {[c.__name__ for c in available_classes]}")
    return cls.from_pretrained(model_path)

@register_to_available_classes
class AutoencoderKLQwenImageAlpha(AutoencoderKLQwenImage):

    @register_to_config
    def __init__(
        self,
        in_channels: int = None,
        out_channels: int = None,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
        latents_std: List[float] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160],
    ):
        super().__init__(
            base_dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            latents_mean=latents_mean,
            latents_std=latents_std,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels is not None:
            self.encoder.conv_in = QwenImageCausalConv3d(
                self.in_channels,
                self.encoder.conv_in.out_channels,
                kernel_size=self.encoder.conv_in.kernel_size,
                stride=self.encoder.conv_in.stride,
                padding=1,
            )
        if self.out_channels is not None:
            self.decoder.conv_out = QwenImageCausalConv3d(
                self.decoder.conv_out.in_channels,
                self.out_channels,
                kernel_size=self.decoder.conv_out.kernel_size,
                stride=self.decoder.conv_out.stride,
                padding=1,
            )