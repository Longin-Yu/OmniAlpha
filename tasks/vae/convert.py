from typing import *

import torch, argparse
import torch.nn as nn
from transformers import AutoModel
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage, QwenImageCausalConv3d

from alpha.vae.modeling import AutoencoderKLQwenImageAlpha, load_vae_from_local_dir

def register_converter(model_type, converter_name):
    def decorator(func):
        setattr(model_type, converter_name, func)
        return func
    return decorator

def replace_conv_in_out(
    model: Union[AutoencoderKLQwenImage, AutoencoderKL], 
    in_channels,
    out_channels,
    padding = None,
):
    conv_in = model.encoder.conv_in
    conv_in_new = conv_in.__class__(
        in_channels,
        conv_in.out_channels,
        conv_in.kernel_size,
        conv_in.stride,
        conv_in.padding if padding is None else padding,
    )
    with torch.no_grad():
        conv_in_new.weight[:, :3] = conv_in.weight
        conv_in_new.weight[:, 3:] = 0       
        conv_in_new.bias.copy_(conv_in.bias)
    model.encoder.conv_in = conv_in_new

    conv_out = model.decoder.conv_out
    conv_out_new = conv_in.__class__(
        conv_out.in_channels,
        out_channels,
        conv_out.kernel_size,
        conv_out.stride,
        conv_out.padding if padding is None else padding,
    )
    with torch.no_grad():
        conv_out_new.weight[:3] = conv_out.weight
        conv_out_new.weight[3:] = 0
        conv_out_new.bias[:3] = conv_out.bias
        conv_out_new.bias[3] = 1
    model.decoder.conv_out = conv_out_new

@register_converter(AutoencoderKLQwenImage, "convert_rgba")
def convert_AutoencoderKLQwenImage(model: AutoencoderKLQwenImage):
    replace_conv_in_out(model, 4, 4, padding=1)

    alpha_model = AutoencoderKLQwenImageAlpha(
        in_channels=4,
        out_channels=4,
        base_dim=model.config.base_dim,
        z_dim=model.config.z_dim,
        dim_mult=model.config.dim_mult,
        num_res_blocks=model.config.num_res_blocks,
        attn_scales=model.config.attn_scales,
        temperal_downsample=model.config.temperal_downsample,
        dropout=model.config.dropout,
        latents_mean=model.config.latents_mean,
        latents_std=model.config.latents_std,
    )
    alpha_model.load_state_dict(model.state_dict())
    return alpha_model


@register_converter(AutoencoderKL, "convert_rgba")
def convert_AutoencoderKL(model: AutoencoderKL):
    replace_conv_in_out(model, 4, 4)
    
    config = dict(model._internal_dict)
    config.update({
        "in_channels": 4,
        "out_channels": 4,
    })
    model._internal_dict = config

    return model

def convert_module(model):
    if hasattr(model, "convert_rgba"):
        return model.convert_rgba()
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--src", type=str, required=True, help="source model path")
    arg_parse.add_argument("--dst", type=str, required=True, help="destination model path")
    args = arg_parse.parse_args()
    
    vae = load_vae_from_local_dir(args.src)
    converted_vae = convert_module(vae)
    converted_vae.save_pretrained(args.dst)

if __name__ == '__main__':
    main()
