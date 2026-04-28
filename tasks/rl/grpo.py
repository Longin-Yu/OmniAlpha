import argparse
import math
import os
from functools import partial
from alpha.grpo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    # nccl_info,
)
from torch.utils.data import DataLoader
import torch
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.distributed import DistributedSampler
# import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from alpha.data import DatasetManager, TI2IDataset
import torch.distributed as dist
from alpha.utils import all_gather_flattened_objects
from alpha.grpo.utils.checkpoint import (
    save_checkpoint,
    # save_lora_checkpoint,
    # resume_lora_optimizer,
)
from alpha.grpo.utils.logging_ import main_print
from diffusers.image_processor import VaeImageProcessor
from alpha.utils.pil import (
    create_image_grid as create_dual_bg_grid,
    resize_image_to_max_pixels,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0") #实际diffuser的版本按omnialpha的来
import time
from collections import deque
import numpy as np
from torch.nn import functional as F
from typing import Any, Dict, List, Optional
from PIL import Image
from alpha.grpo.utils.fsdp_util_qwenimage import fsdp_wrapper, FSDPConfig, apply_fsdp_checkpointing
from contextlib import contextmanager
from safetensors.torch import save_file

from peft import LoraConfig, get_peft_model_state_dict
from alpha.pipelines.qwen_image_edit import CustomQwenImageEditPlusPipeline as QwenImageEditPlusPipeline 
from alpha.pipelines.qwen_image_edit import QwenImageEditModules
from alpha.vae.modeling import load_vae_from_local_dir

    
class FSDP_EMA:
    def __init__(self, model, decay, rank):
        self.decay = decay
        self.rank = rank
        self.ema_state_dict_rank0 = {}
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            self.ema_state_dict_rank0 = {k: v.clone() for k, v in state_dict.items()}
            main_print("--> Modern EMA handler initialized on rank 0.")

    def update(self, model):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            for key in self.ema_state_dict_rank0:
                if key in model_state_dict:
                    self.ema_state_dict_rank0[key].copy_(
                        self.decay * self.ema_state_dict_rank0[key] + (1 - self.decay) * model_state_dict[key]
                    )

    @contextmanager
    def use_ema_weights(self, model):
        backup_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        backup_state_dict_rank0 = get_model_state_dict(model, options=backup_options)

        load_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_model_state_dict(
            model,
            model_state_dict=self.ema_state_dict_rank0, 
            options=load_options
        )
        
        try:
            yield
        finally:
            restore_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            set_model_state_dict(
                model,
                model_state_dict=backup_state_dict_rank0, 
                options=restore_options
            )

def save_ema_checkpoint(ema_handler, rank, output_dir, step, epoch, config_dict):
    if rank == 0 and ema_handler is not None:
        ema_checkpoint_path = os.path.join(output_dir, f"checkpoint-ema-{step}-{epoch}")
        os.makedirs(ema_checkpoint_path, exist_ok=True)
        weight_path = os.path.join(ema_checkpoint_path ,
                                   "diffusion_pytorch_model.safetensors")
        save_file(ema_handler.ema_state_dict_rank0, weight_path)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        config_path = os.path.join(ema_checkpoint_path, "config.json")
        # save dict as json
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        #torch.save(ema_handler.ema_state_dict_rank0, os.path.join(ema_checkpoint_path, "ema_model.pt"))
        main_print(f"--> EMA checkpoint saved at {ema_checkpoint_path}")


def sanitize_filename(name: Optional[Any], max_length: int = 96) -> str:
    if not name:
        return "sample"
    name = str(name)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name.strip())
    safe = safe.strip("._") or "sample"
    return safe[:max_length]


def vis_data_to_grid_image(vis_data_list: List[dict], gap: int = 12) -> Optional[Image.Image]:
    rows: List[List[List[Image.Image]]] = []
    for vis_data in vis_data_list:
        row_groups: List[List[Image.Image]] = []
        for key in ["input_images", "output_images", "predictions"]:
            group_images: List[Image.Image] = []
            for img in vis_data.get(key, []):
                if not isinstance(img, Image.Image):
                    raise TypeError(f"{key} must contain PIL images, got {type(img)}")
                group_images.append(img.convert("RGBA"))
            if key == "predictions" and vis_data.get("blended_prediction") is not None:
                blended_prediction = vis_data["blended_prediction"]
                if not isinstance(blended_prediction, Image.Image):
                    raise TypeError(f"blended_prediction must be a PIL image, got {type(blended_prediction)}")
                group_images.append(blended_prediction.convert("RGBA"))
            if group_images:
                row_groups.append(group_images)
        if row_groups:
            rows.append(row_groups)
    return create_dual_bg_grid(rows, gap=gap) if rows else None


def pil_image_to_chw_array(image: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.array(image.convert("RGB")).transpose(2, 0, 1))


def save_vis_grid(save_dir: str, global_step: int, vis_data_list: List[dict], gap: int = 12) -> Optional[Image.Image]:
    os.makedirs(save_dir, exist_ok=True)
    grid_image = vis_data_to_grid_image(vis_data_list, gap=gap)
    if grid_image is None:
        return None
    filename = f"grid_{global_step:06d}.png"
    grid_image.save(os.path.join(save_dir, filename))
    return grid_image


def save_vis_single_images(save_dir: str, global_step: int, vis_data_list: List[dict]):
    step_dir = os.path.join(save_dir, f"step_{global_step:06d}")
    os.makedirs(step_dir, exist_ok=True)

    for idx, vis_data in enumerate(vis_data_list):
        sample_key = vis_data.get("sample_key") or f"sample_{idx:03d}"
        sample_dir = os.path.join(step_dir, f"{idx:03d}_{sanitize_filename(sample_key)}")
        os.makedirs(sample_dir, exist_ok=True)

        prompt = vis_data.get("prompt")
        if prompt:
            with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
                f.write(prompt)

        with open(os.path.join(sample_dir, "meta.txt"), "w") as f:
            f.write(f"sample_key: {sample_key}\n")
            f.write(f"begin_with_bg: {bool(vis_data.get('begin_with_bg', False))}\n")

        for k, img in enumerate(vis_data.get("input_images", [])):
            img.save(os.path.join(sample_dir, f"input_{k:02d}.png"))

        for k, img in enumerate(vis_data.get("output_images", [])):
            img.save(os.path.join(sample_dir, f"output_{k:02d}.png"))

        for k, img in enumerate(vis_data.get("predictions", [])):
            img.save(os.path.join(sample_dir, f"prediction_{k:02d}.png"))

        blended_prediction = vis_data.get("blended_prediction")
        if blended_prediction is not None:
            blended_prediction.save(os.path.join(sample_dir, "blended_prediction.png"))


def log_vis_to_tensorboard(
    writer: Optional[SummaryWriter],
    global_step: int,
    vis_data_list: List[dict],
    grid_image: Optional[Image.Image],
):
    if writer is None or grid_image is None:
        return

    tb_grid_image = resize_image_to_max_pixels(grid_image, 8192 ** 2)
    writer.add_image("train-vis/image_grid", pil_image_to_chw_array(tb_grid_image), global_step=global_step)

    prompt_lines = []
    for idx, vis_data in enumerate(vis_data_list):
        prompt_lines.append(
            f"Item {idx} | sample_key={vis_data.get('sample_key', '')} | "
            f"begin_with_bg={bool(vis_data.get('begin_with_bg', False))}\n"
            f"{vis_data.get('prompt', '')}"
        )
    if prompt_lines:
        writer.add_text("train-vis/prompts", "\n\n".join(prompt_lines), global_step=global_step)


def add_scalar_dict(writer: Optional[SummaryWriter], global_step: int, scalars: Dict[str, float]):
    if writer is None:
        return
    for key, value in scalars.items():
        if value is None:
            continue
        writer.add_scalar(key, float(value), global_step)
        
        
def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)
    

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t 
        

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def raw_ti2i_collate_function(examples):
    if len(examples) != 1:
        raise ValueError("Raw GRPO data loading currently supports train_batch_size=1 only.")

    example = examples[0]
    batch = {
        "prompt": [example["prompt"]],
        "output_images": [example["output_images"]],
        "output_image_paths": [example.get("output_image_paths", [])],
        "sample_keys": [example.get("sample_key")],
        "begin_with_bg": [bool(example.get("begin_with_bg", False))],
    }
    if "input_images" in example:
        batch["input_images"] = [example["input_images"]]
        batch["input_image_paths"] = [example.get("input_image_paths", [])]
    return batch


def split_layers_by_begin_with_bg(layers: List, begin_with_bg: bool):
    if begin_with_bg and len(layers) > 0:
        return layers[0], layers[1:]
    return None, list(layers)


def infer_num_output_frames(output_images: List[Image.Image], sample_key: Optional[Any] = None) -> int:
    frames = len(output_images)
    if frames <= 0:
        key = sample_key if sample_key is not None else "unknown"
        raise ValueError(f"Sample {key} has no output_images to infer output frame count.")
    return frames


def infer_chunk_num_output_frames(batch_examples: List[dict]) -> int:
    if not batch_examples:
        raise ValueError("batch_examples must not be empty when inferring output frame count.")
    frame_counts = [
        infer_num_output_frames(example["output_images"], example.get("sample_key"))
        for example in batch_examples
    ]
    frames = frame_counts[0]
    if any(count != frames for count in frame_counts[1:]):
        raise ValueError(f"All samples in one GRPO chunk must share the same output frame count, got {frame_counts}.")
    return frames


def pil_rgba_to_tensor01(image: Image.Image, device):
    rgba = image.convert("RGBA")
    arr = np.array(rgba, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return x.to(device=device, dtype=torch.float32)


def prepare_online_grpo_batch(batch, pipeline, device, vae_dtype):
    prompts = batch["prompt"]
    if len(prompts) != 1:
        raise ValueError("Raw GRPO online preprocessing currently supports train_batch_size=1 only.")

    input_images = batch.get("input_images", [[]])[0]

    with torch.inference_mode():
        if input_images:
            input_prompt_images, _ = pipeline.prepare_images(
                input_images,
                "condition",
                reshape=False,
            )
            prompt_embeds, prompt_attention_mask = pipeline.encode_prompt(
                prompt=prompts,
                image=input_prompt_images,
                device=device,
            )

            input_vae_images, _ = pipeline.prepare_images(
                input_images,
                "vae",
                reshape=False,
            )
            with torch.autocast("cuda", dtype=vae_dtype):
                image_latents_list = [
                    pipeline._encode_vae_image(img.to(device=device, dtype=vae_dtype), generator=None)
                    for img in input_vae_images
                ]
            image_latents = torch.cat(image_latents_list, dim=2)
        else:
            prompt_embeds, prompt_attention_mask = pipeline.encode_prompt(
                prompt=prompts,
                image=None,
                device=device,
            )
            output_images = batch["output_images"][0]
            if not output_images:
                raise ValueError("Sample has neither input_images nor output_images.")
            ref_vae_images, _ = pipeline.prepare_images(
                [output_images[0]],
                "vae",
                reshape=False,
            )
            with torch.autocast("cuda", dtype=vae_dtype):
                ref_latents = pipeline._encode_vae_image(
                    ref_vae_images[0].to(device=device, dtype=vae_dtype),
                    generator=None,
                )
            image_latents = ref_latents.new_empty(
                (
                    ref_latents.shape[0],
                    ref_latents.shape[1],
                    0,
                    ref_latents.shape[-2],
                    ref_latents.shape[-1],
                )
            )

    original_length = prompt_attention_mask.sum(dim=1).to(dtype=torch.long).cpu()
    batch_examples = []
    for idx, prompt in enumerate(prompts):
        batch_examples.append(
            {
                "prompt": prompt,
                "sample_key": batch["sample_keys"][idx],
                "begin_with_bg": bool(batch["begin_with_bg"][idx]),
                "input_images": batch.get("input_images", [[]])[idx] if "input_images" in batch else [],
                "output_images": batch["output_images"][idx],
                "input_image_paths": batch.get("input_image_paths", [[]])[idx] if "input_image_paths" in batch else [],
                "output_image_paths": batch["output_image_paths"][idx],
            }
        )
    return prompt_embeds, prompt_attention_mask, image_latents, prompts, original_length, batch_examples


def pred_rgba_to_01(pred_rgba: torch.Tensor):
    rgb = torch.clamp((pred_rgba[:3] + 1.0) / 2.0, 0.0, 1.0)
    a = torch.clamp((pred_rgba[3:4] + 1.0) / 2.0, 0.0, 1.0)
    return torch.cat([rgb, a], dim=0)

def rgba01_tensor_to_pil(rgba01: torch.Tensor) -> Image.Image:
    """
    rgba01: (4, H, W), each channel in [0,1]
    return: PIL RGBA
    """
    x = torch.clamp(rgba01, 0.0, 1.0).detach().cpu()
    x = (x * 255.0).round().to(torch.uint8)
    arr = x.permute(1, 2, 0).contiguous().numpy()  # H,W,4
    return Image.fromarray(arr, mode="RGBA")

def resize_chw(x: torch.Tensor, out_hw):
    h, w = out_hw
    if x.shape[-2:] == (h, w):
        return x
    return F.interpolate(x.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


def blend_rgba_on_bg01(rgba01: torch.Tensor, bg_rgb: torch.Tensor):
    rgb = rgba01[:3]
    a = rgba01[3:4]
    return rgb * a + bg_rgb * (1.0 - a)


def alpha_compose_rgba01(layers_rgba01: List[torch.Tensor]):
    eps = 1e-6
    comp_rgb = torch.zeros_like(layers_rgba01[0][:3])
    comp_a = torch.zeros_like(layers_rgba01[0][3:4])
    for lay in layers_rgba01:
        rgb, a = lay[:3], lay[3:4]
        new_a = a + comp_a * (1.0 - a)
        new_rgb = (rgb * a + comp_rgb * comp_a * (1.0 - a)) / torch.clamp(new_a, eps, 1.0)
        comp_rgb, comp_a = new_rgb, new_a
    return torch.cat([comp_rgb, comp_a], dim=0)


def ssim_score(pred_rgb01: torch.Tensor, gt_rgb01: torch.Tensor, ssim_loss):
    loss = ssim_loss(pred_rgb01.unsqueeze(0), gt_rgb01.unsqueeze(0))
    if loss.ndim > 0:
        loss = loss.view(-1).mean()
    return 1.0 - loss


def psnr_reward(pred_rgb01: torch.Tensor, gt_rgb01: torch.Tensor, max_db: float):
    mse = torch.mean((pred_rgb01 - gt_rgb01) ** 2)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    return torch.clamp(psnr / max_db, 0.0, 1.0)


def lpips_reward_score(pred_rgb01: torch.Tensor, gt_rgb01: torch.Tensor, lpips_loss):
    p = pred_rgb01.unsqueeze(0) * 2.0 - 1.0
    g = gt_rgb01.unsqueeze(0) * 2.0 - 1.0
    lp = lpips_loss(p, g).view(-1).mean()
    return 1.0 - lp


def gaussian_kernel2d(radius: int, device, dtype=torch.float32):
    if radius <= 0:
        return None
    sigma = max(radius / 3.0, 1e-6)
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g)


def boundary_alpha_l1_reward(pred_a01: torch.Tensor, gt_a01: torch.Tensor, low: float, high: float, blur_px: int):
    pred_b = ((pred_a01 >= low) & (pred_a01 <= high)).float()
    gt_b = ((gt_a01 >= low) & (gt_a01 <= high)).float()
    if blur_px > 0:
        k = gaussian_kernel2d(blur_px, device=pred_a01.device, dtype=pred_a01.dtype)
        k = k.unsqueeze(0).unsqueeze(0)
        pred_w = F.conv2d(pred_b.unsqueeze(0), k, padding=blur_px).squeeze(0)
        gt_w = F.conv2d(gt_b.unsqueeze(0), k, padding=blur_px).squeeze(0)
    else:
        pred_w, gt_w = pred_b, gt_b
    w = torch.clamp(pred_w + gt_w, 0.0, 1.0)
    l1 = (w * torch.abs(pred_a01 - gt_a01)).sum() / (w.sum() + 1e-6)
    return torch.clamp(1.0 - l1, 0.0, 1.0)

def run_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        prompt_attention_mask,
        img_shapes,
        txt_seq_lens,
        image_latents,
        grpo_sample,
    ):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            transformer.eval()
            latent_model_input = torch.cat([z, image_latents], dim=1)
            with torch.autocast("cuda", torch.bfloat16):
                pred= transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            pred = pred[:, :z.shape[1]]
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        latents = pred_original
        all_latents = torch.stack(all_latents, dim=1)  
        all_log_probs = torch.stack(all_log_probs, dim=1)  
        return z, latents, all_latents, all_log_probs

        
def grpo_one_step(
        args,
        latents,
        pre_latents,
        encoder_hidden_states, 
        prompt_attention_masks, 
        txt_seq_lens,
        img_shapes,
        image_latents,
        transformer,
        timesteps,
        i,
        sigma_schedule,
):
    B = encoder_hidden_states.shape[0]
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        pred= transformer(
            hidden_states=latent_model_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_attention_masks,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        pred = pred[:, :latents.shape[1]]
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob


def sample_reference_model(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    prompt_attention_masks, 
    original_length,
    image_latents, 
    batch_examples,
    # reward_model,
    # tokenizer,
    prompt,
    # preprocess_val,
    lpips_loss = None,
    ssim_loss = None,
    do_visualize: bool = False,
):
    vis_data = None
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
        
    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0] 
    batch_size = args.num_generations if (args.use_group and args.init_same_noise) else 1
    # batch_size = 1 
    batch_indices = torch.chunk(torch.arange(B), B // batch_size) 

    all_latents = []
    all_log_probs = []
    
    all_reward_r1_ssim = []
    all_reward_r1_psnr = []
    all_reward_r2_ssim = []
    all_reward_r2_psnr = []
    all_reward_r2_lpips = []
    all_reward_r3_bg_lpips = []
    all_reward_r4_fg_boundary = []
    all_reward_final = [] 
    all_valid_r1_ssim = []
    all_valid_r1_psnr = []
    all_valid_r2_ssim = []
    all_valid_r2_psnr = []
    all_valid_r2_lpips = []
    all_valid_r3_bg_lpips = []
    all_valid_r4_fg_boundary = []
    
    all_txt_seq_lens = []
    all_image_latents = []
    all_img_shapes = []
    all_num_output_frames = []
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # if args.init_same_noise:
    #     input_latents = torch.randn(
    #             (1, 1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
    #             device=device,
    #             dtype=torch.bfloat16,
    #         ) 
    
    
    for index, batch_idx in enumerate(batch_indices):
        chunk_original_length = int(original_length[batch_idx[0]].item())
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx][:, :chunk_original_length]
        batch_prompt_attention_mask = prompt_attention_masks[batch_idx][:, :chunk_original_length]
        batch_image_latents = image_latents[batch_idx]  #batch_image_latents.shape=(1, z_dim, 1, H, W) 
        batch_prompt = [prompt[i.item()] for i in batch_idx]
        batch_examples_for_chunk = [batch_examples[i.item()] for i in batch_idx]
        frames = infer_chunk_num_output_frames(batch_examples_for_chunk)
        # if not args.init_same_noise: 
            # input_latents = torch.randn(
            #         (len(batch_idx), 1, IN_CHANNELS, latent_h, latent_w),  #（c,t,h,w)
            #         device=device,
            #         dtype=torch.bfloat16,
            #     ) 
            
        _, z_dim, F_cond, H_cond, W_cond = batch_image_latents.shape 
        latent_h = H_cond
        latent_w = W_cond
        
        
        if args.init_same_noise:
            base = torch.randn((1, vae.config.z_dim, frames, latent_h, latent_w), device=device, dtype=torch.bfloat16)
            output_latents = base.repeat(len(batch_idx), 1, 1, 1, 1)
        else:
            output_latents = torch.randn((len(batch_idx), vae.config.z_dim, frames, latent_h, latent_w), device=device, dtype=torch.bfloat16)
            
         
        # packed_height = 2 * (int(h) // (8 * 2))
        # packed_width = 2 * (int(w) // (8 * 2)) 
        
        
        # packed_height = 2 * (int(h) // (8 * 2))
        # packed_width = 2 * (int(w) // (8 * 2)) 
        
        # input_latents_new = pack_latents(input_latents, len(batch_idx), 16, packed_height, packed_width)   
        image_latents_packed = QwenImageEditPlusPipeline.pack_latents_multi_frames(
            batch_image_latents
        ) 
        
        txt_seq_lens =  batch_prompt_attention_mask.sum(dim=1).tolist() 
        
        # img_shapes = [[
        #     (1, h // 8 // 2, w // 8// 2),  # 输出图的 latent patch 尺寸
        #     (1, batch_calculated_height.item() // 8 // 2, batch_calculated_width.item() // 8 // 2), # 条件图 latent patch 尺寸
        #     ]] 
  
        img_shapes = [[
            *[(1, latent_h // 2, latent_w // 2) for _ in range(frames)],   # 输出帧
            *[(1, H_cond // 2, W_cond // 2) for _ in range(F_cond)],       # 条件帧
        ] for _ in range(len(batch_idx))] 
        
        
        for _ in range(len(batch_idx)):
            all_img_shapes.append(img_shapes[0])
            all_num_output_frames.append(frames)
            
        grpo_sample=True
        progress_bar = tqdm(
            range(0, sample_steps),
            desc="Sampling Progress",
            disable=rank != 0,
            leave=False,
        )
        # image_latent_height, image_latent_width = batch_image_latents.shape[3:]
        # image_latents_new = pack_latents(
        #         batch_image_latents, len(batch_idx), 16, image_latent_height, image_latent_width
        #     )
        
        output_latents_packed = QwenImageEditPlusPipeline.pack_latents_multi_frames(
            output_latents 
        )  
       
        with torch.no_grad():
            z, latents_packed, batch_latents, batch_log_probs = run_sample_step(
                args,
                output_latents_packed.clone(),
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_prompt_attention_mask,
                img_shapes,
                txt_seq_lens,
                image_latents_packed,
                grpo_sample,
            )
       
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_txt_seq_lens.append(torch.tensor(txt_seq_lens))
        all_image_latents.append(image_latents_packed)
        
        vae.enable_tiling()
        vae_decode_dtype = next(vae.parameters()).dtype
        
        image_processor = VaeImageProcessor(16)

        
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # latents = unpack_latents(latents, h, w, 8)
                
                latents = QwenImageEditPlusPipeline.unpack_latents_multi_frames(
                    latents_packed,   
                    frames,
                    latent_h,
                    latent_w,
                )  # B=1, C=16, F=2, H, W
                
                latents = latents.permute(0, 2, 1, 3, 4) # B, F, C, H, W
                B1, F1, C1, H1, W1 = latents.shape
                latents = latents.view(B1 * F1, C1, 1, H1, W1)
                latents = latents.to(vae_decode_dtype)
                
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, vae_decode_dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                    latents.device, vae_decode_dtype
                )
                latents = (latents / latents_std + latents_mean).to(vae_decode_dtype)
                
                decoded = vae.decode(latents, return_dict=False)[0][:, :, 0]

                H1, W1 = decoded.shape[-2:]
                images = decoded.view(B1, F1, 4, H1, W1).permute(0, 2, 1, 3, 4)
                
        
        # images: (bs, 4, frames, H, W)  
        layers = [images[:, :, k] for k in range(frames)]            # list of (bs,4,H,W)

        # ===== 统一转到 [0,1] RGBA 空间，再做 alpha compose =====
        layers_01 = []
        for lay in layers:
            rgb01 = torch.clamp((lay[:, :3] + 1.0) / 2.0, 0.0, 1.0)
            a01   = torch.clamp((lay[:, 3:4] + 1.0) / 2.0, 0.0, 1.0)
            lay01 = torch.cat([rgb01, a01], dim=1)   # (bs,4,H,W)
            layers_01.append(lay01)

        comp_list = []
        for bi in range(B1):
            comp_list.append(
                alpha_compose_rgba01([layers_01[k][bi] for k in range(frames)])
            )

        I_pred_01 = torch.stack(comp_list, dim=0)   # (bs,4,H,W), all in [0,1]
        
        
        # ====== 训练中可视化：只在需要的时候、只抽当前 batch 的第一个样本 ======
        if do_visualize and vis_data is None:
            idx = 0
            blend_rgba_pil = rgba01_tensor_to_pil(I_pred_01[idx].detach().cpu())
            prediction_pils = [
                image_processor.postprocess(layers[k][idx:idx+1].detach().cpu())[0]
                for k in range(frames)
            ]

            vis_example = batch_examples_for_chunk[idx]
            vis_data = {
                "begin_with_bg": bool(vis_example["begin_with_bg"]),
                "sample_key": vis_example.get("sample_key"),
                "prompt": batch_prompt[idx],
                "input_images": list(vis_example.get("input_images", [])),
                "output_images": list(vis_example["output_images"]),
                "predictions": prediction_pils,
                "blended_prediction": blend_rgba_pil if bool(vis_example["begin_with_bg"]) else None,
            }
        # ====== 可视化片段结束 ======


        # ---------- new rewards ----------
        Bcur = len(batch_examples_for_chunk)
        w_white = torch.ones((3, 1, 1), device=device, dtype=torch.float32)
        w_black = torch.zeros((3, 1, 1), device=device, dtype=torch.float32)

        r1_ssim_vals = []
        r1_psnr_vals = []
        r2_ssim_vals = []
        r2_psnr_vals = []
        r2_lpips_vals = []
        r3_bg_lpips_vals = []
        r4_fg_boundary_vals = []
        final_vals = []
        valid_r1_ssim_vals = []
        valid_r1_psnr_vals = []
        valid_r2_ssim_vals = []
        valid_r2_psnr_vals = []
        valid_r2_lpips_vals = []
        valid_r3_bg_lpips_vals = []
        valid_r4_fg_boundary_vals = []

        for bi in range(Bcur):
            example = batch_examples_for_chunk[bi]
            key = example.get("sample_key", f"sample_{bi}")
            pred_layers = [pred_rgba_to_01(layers[k][bi].to(torch.float32)) for k in range(frames)]
            gt_output_layers = [
                pil_rgba_to_tensor01(img, device)
                for img in example["output_images"]
            ]
            bg_gt, fg_gt_list = split_layers_by_begin_with_bg(
                gt_output_layers,
                example["begin_with_bg"],
            )
            pred_bg, pred_fg_list = split_layers_by_begin_with_bg(
                pred_layers,
                example["begin_with_bg"],
            )
            whole_input_images = example.get("input_images", [])
            if whole_input_images:
                whole_gt = pil_rgba_to_tensor01(whole_input_images[0], device)
            elif gt_output_layers:
                whole_gt = alpha_compose_rgba01(gt_output_layers)
            else:
                raise ValueError(f"Sample {key} has no GT images to compute rewards.")

            has_bg = bg_gt is not None
            has_fg = len(fg_gt_list) > 0
            gt_layers = ([bg_gt] if bg_gt is not None else []) + fg_gt_list
            pred_layers_for_r1 = ([pred_bg] if pred_bg is not None else []) + pred_fg_list

            layer_ssim_scores = []
            layer_psnr_scores = []
            n_match = min(len(pred_layers_for_r1), len(gt_layers))
            for li in range(n_match):
                pred_l = resize_chw(pred_layers_for_r1[li], gt_layers[li].shape[-2:])
                gt_l = gt_layers[li]
                pred_w = blend_rgba_on_bg01(pred_l, w_white)
                pred_b = blend_rgba_on_bg01(pred_l, w_black)
                gt_w = blend_rgba_on_bg01(gt_l, w_white)
                gt_b = blend_rgba_on_bg01(gt_l, w_black)

                if args.use_r1_layer_blend_ssim:
                    assert ssim_loss is not None, "r1 SSIM enabled but ssim_loss is None"
                    s_w = ssim_score(pred_w, gt_w, ssim_loss)
                    s_b = ssim_score(pred_b, gt_b, ssim_loss)
                    layer_ssim_scores.append(0.5 * (s_w + s_b))

                if args.use_r1_layer_blend_psnr:
                    p_w = psnr_reward(pred_w, gt_w, args.psnr_max_db)
                    p_b = psnr_reward(pred_b, gt_b, args.psnr_max_db)
                    layer_psnr_scores.append(0.5 * (p_w + p_b))

            r1_ssim = torch.stack(layer_ssim_scores).mean() if layer_ssim_scores else torch.tensor(0.0, device=device)
            r1_psnr = torch.stack(layer_psnr_scores).mean() if layer_psnr_scores else torch.tensor(0.0, device=device)
            valid_r1_ssim = len(layer_ssim_scores) > 0
            valid_r1_psnr = len(layer_psnr_scores) > 0

            valid_r2 = has_bg and has_fg and pred_bg is not None and len(pred_fg_list) > 0
            if valid_r2:
                composed_pred = alpha_compose_rgba01([pred_bg] + pred_fg_list)
                composed_pred = resize_chw(composed_pred, whole_gt.shape[-2:])
                pred_w = blend_rgba_on_bg01(composed_pred, w_white)
                pred_b = blend_rgba_on_bg01(composed_pred, w_black)
                gt_w = blend_rgba_on_bg01(whole_gt, w_white)
                gt_b = blend_rgba_on_bg01(whole_gt, w_black)

                if args.use_r2_comp_ssim:
                    assert ssim_loss is not None, "r2 SSIM enabled but ssim_loss is None"
                    r2_ssim = 0.5 * (ssim_score(pred_w, gt_w, ssim_loss) + ssim_score(pred_b, gt_b, ssim_loss))
                else:
                    r2_ssim = torch.tensor(0.0, device=device)
                if args.use_r2_comp_psnr:
                    r2_psnr = 0.5 * (psnr_reward(pred_w, gt_w, args.psnr_max_db) + psnr_reward(pred_b, gt_b, args.psnr_max_db))
                else:
                    r2_psnr = torch.tensor(0.0, device=device)
                if args.use_r2_comp_lpips:
                    assert lpips_loss is not None, "r2 LPIPS enabled but lpips_loss is None"
                    r2_lpips = 0.5 * (lpips_reward_score(pred_w, gt_w, lpips_loss) + lpips_reward_score(pred_b, gt_b, lpips_loss))
                else:
                    r2_lpips = torch.tensor(0.0, device=device)
            else:
                r2_ssim = torch.tensor(0.0, device=device)
                r2_psnr = torch.tensor(0.0, device=device)
                r2_lpips = torch.tensor(0.0, device=device)

            valid_r3 = has_bg and pred_bg is not None
            if valid_r3 and args.use_r3_bg_blend_lpips:
                assert lpips_loss is not None, "r3 LPIPS enabled but lpips_loss is None"
                pred_bg_rs = resize_chw(pred_bg, bg_gt.shape[-2:])
                pred_w = blend_rgba_on_bg01(pred_bg_rs, w_white)
                pred_b = blend_rgba_on_bg01(pred_bg_rs, w_black)
                gt_w = blend_rgba_on_bg01(bg_gt, w_white)
                gt_b = blend_rgba_on_bg01(bg_gt, w_black)
                r3_bg_lpips = 0.5 * (lpips_reward_score(pred_w, gt_w, lpips_loss) + lpips_reward_score(pred_b, gt_b, lpips_loss))
            else:
                r3_bg_lpips = torch.tensor(0.0, device=device)

            valid_r4 = has_fg and len(pred_fg_list) > 0
            if valid_r4 and args.use_r4_fg_boundary_alpha_l1:
                n_fg = min(len(pred_fg_list), len(fg_gt_list))
                fg_rewards = []
                for fi in range(n_fg):
                    pred_fg = resize_chw(pred_fg_list[fi], fg_gt_list[fi].shape[-2:])
                    gt_fg = fg_gt_list[fi]
                    fg_rewards.append(
                        boundary_alpha_l1_reward(
                            pred_fg[3:4],
                            gt_fg[3:4],
                            args.boundary_alpha_low,
                            args.boundary_alpha_high,
                            args.boundary_blur_px,
                        )
                    )
                r4_fg_boundary = torch.stack(fg_rewards).mean() if fg_rewards else torch.tensor(0.0, device=device)
            else:
                r4_fg_boundary = torch.tensor(0.0, device=device)

            reward_terms = []
            reward_weights = []
            if args.use_r1_layer_blend_ssim and args.w_r1_layer_blend_ssim > 0 and valid_r1_ssim:
                reward_terms.append(r1_ssim)
                reward_weights.append(args.w_r1_layer_blend_ssim)
            if args.use_r1_layer_blend_psnr and args.w_r1_layer_blend_psnr > 0 and valid_r1_psnr:
                reward_terms.append(r1_psnr)
                reward_weights.append(args.w_r1_layer_blend_psnr)
            if args.use_r2_comp_ssim and args.w_r2_comp_ssim > 0 and valid_r2:
                reward_terms.append(r2_ssim)
                reward_weights.append(args.w_r2_comp_ssim)
            if args.use_r2_comp_psnr and args.w_r2_comp_psnr > 0 and valid_r2:
                reward_terms.append(r2_psnr)
                reward_weights.append(args.w_r2_comp_psnr)
            if args.use_r2_comp_lpips and args.w_r2_comp_lpips > 0 and valid_r2:
                reward_terms.append(r2_lpips)
                reward_weights.append(args.w_r2_comp_lpips)
            if args.use_r3_bg_blend_lpips and args.w_r3_bg_blend_lpips > 0 and valid_r3:
                reward_terms.append(r3_bg_lpips)
                reward_weights.append(args.w_r3_bg_blend_lpips)
            if args.use_r4_fg_boundary_alpha_l1 and args.w_r4_fg_boundary_alpha_l1 > 0 and valid_r4:
                reward_terms.append(r4_fg_boundary)
                reward_weights.append(args.w_r4_fg_boundary_alpha_l1)

            if len(reward_terms) == 0:
                raise ValueError("No active reward term for current sample. Check reward switches and GT routing.")
            w = torch.tensor(reward_weights, device=device, dtype=torch.float32)
            t = torch.stack([x.to(torch.float32) for x in reward_terms], dim=0)
            final = (w * t).sum() / (w.sum() + 1e-8)

            r1_ssim_vals.append(r1_ssim.to(torch.float32))
            r1_psnr_vals.append(r1_psnr.to(torch.float32))
            r2_ssim_vals.append(r2_ssim.to(torch.float32))
            r2_psnr_vals.append(r2_psnr.to(torch.float32))
            r2_lpips_vals.append(r2_lpips.to(torch.float32))
            r3_bg_lpips_vals.append(r3_bg_lpips.to(torch.float32))
            r4_fg_boundary_vals.append(r4_fg_boundary.to(torch.float32))
            final_vals.append(final.to(torch.float32))
            valid_r1_ssim_vals.append(valid_r1_ssim)
            valid_r1_psnr_vals.append(valid_r1_psnr)
            valid_r2_ssim_vals.append(valid_r2)
            valid_r2_psnr_vals.append(valid_r2)
            valid_r2_lpips_vals.append(valid_r2)
            valid_r3_bg_lpips_vals.append(valid_r3)
            valid_r4_fg_boundary_vals.append(valid_r4)

        all_reward_r1_ssim.append(torch.stack(r1_ssim_vals, dim=0))
        all_reward_r1_psnr.append(torch.stack(r1_psnr_vals, dim=0))
        all_reward_r2_ssim.append(torch.stack(r2_ssim_vals, dim=0))
        all_reward_r2_psnr.append(torch.stack(r2_psnr_vals, dim=0))
        all_reward_r2_lpips.append(torch.stack(r2_lpips_vals, dim=0))
        all_reward_r3_bg_lpips.append(torch.stack(r3_bg_lpips_vals, dim=0))
        all_reward_r4_fg_boundary.append(torch.stack(r4_fg_boundary_vals, dim=0))
        all_reward_final.append(torch.stack(final_vals, dim=0))
        all_valid_r1_ssim.append(torch.tensor(valid_r1_ssim_vals, device=device, dtype=torch.bool))
        all_valid_r1_psnr.append(torch.tensor(valid_r1_psnr_vals, device=device, dtype=torch.bool))
        all_valid_r2_ssim.append(torch.tensor(valid_r2_ssim_vals, device=device, dtype=torch.bool))
        all_valid_r2_psnr.append(torch.tensor(valid_r2_psnr_vals, device=device, dtype=torch.bool))
        all_valid_r2_lpips.append(torch.tensor(valid_r2_lpips_vals, device=device, dtype=torch.bool))
        all_valid_r3_bg_lpips.append(torch.tensor(valid_r3_bg_lpips_vals, device=device, dtype=torch.bool))
        all_valid_r4_fg_boundary.append(torch.tensor(valid_r4_fg_boundary_vals, device=device, dtype=torch.bool))


    rewards_dict = {
        "r1_layer_blend_ssim": torch.cat(all_reward_r1_ssim, dim=0),
        "r1_layer_blend_psnr": torch.cat(all_reward_r1_psnr, dim=0),
        "r2_comp_ssim": torch.cat(all_reward_r2_ssim, dim=0),
        "r2_comp_psnr": torch.cat(all_reward_r2_psnr, dim=0),
        "r2_comp_lpips": torch.cat(all_reward_r2_lpips, dim=0),
        "r3_bg_blend_lpips": torch.cat(all_reward_r3_bg_lpips, dim=0),
        "r4_fg_boundary_alpha_l1": torch.cat(all_reward_r4_fg_boundary, dim=0),
        "final": torch.cat(all_reward_final, dim=0),
    }
    reward_valid_masks = {
        "r1_layer_blend_ssim": torch.cat(all_valid_r1_ssim, dim=0),
        "r1_layer_blend_psnr": torch.cat(all_valid_r1_psnr, dim=0),
        "r2_comp_ssim": torch.cat(all_valid_r2_ssim, dim=0),
        "r2_comp_psnr": torch.cat(all_valid_r2_psnr, dim=0),
        "r2_comp_lpips": torch.cat(all_valid_r2_lpips, dim=0),
        "r3_bg_blend_lpips": torch.cat(all_valid_r3_bg_lpips, dim=0),
        "r4_fg_boundary_alpha_l1": torch.cat(all_valid_r4_fg_boundary, dim=0),
    }
    
    # all_rewards = torch.cat(all_rewards, dim=0)
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_txt_seq_lens = torch.cat(all_txt_seq_lens, dim=0)
    all_image_latents = torch.cat(all_image_latents, dim=0)

    return (
        rewards_dict,
        reward_valid_masks,
        all_latents,
        all_log_probs,
        sigma_schedule,
        all_txt_seq_lens,
        all_image_latents,
        all_img_shapes,
        all_num_output_frames,
        vis_data,
    )
    


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def add_tensor_stats(metrics: Dict[str, float], prefix: str, tensor: torch.Tensor):
    tensor = tensor.detach().to(torch.float32)
    metrics[f"{prefix}/mean"] = tensor.mean().item()
    metrics[f"{prefix}/std"] = tensor.std(unbiased=False).item()
    metrics[f"{prefix}/min"] = tensor.min().item()
    metrics[f"{prefix}/max"] = tensor.max().item()


def add_masked_tensor_stats(
    metrics: Dict[str, float],
    prefix: str,
    tensor: torch.Tensor,
    valid_mask: torch.Tensor,
):
    gathered_tensor = gather_tensor(tensor.detach().to(torch.float32))
    gathered_mask = gather_tensor(valid_mask.detach().to(torch.float32)) > 0.5
    valid_tensor = gathered_tensor[gathered_mask]
    if valid_tensor.numel() == 0:
        return
    add_tensor_stats(metrics, prefix, valid_tensor)


def masked_zscore_by_group(x: torch.Tensor, valid_mask: torch.Tensor, G: int):
    B = x.numel()
    assert B % G == 0
    out = torch.zeros_like(x)
    out_valid_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    for gi in range(B // G):
        s = gi * G
        e = (gi + 1) * G
        group_mask = valid_mask[s:e]
        if not torch.any(group_mask):
            continue
        group_x = x[s:e]
        valid_group_x = group_x[group_mask]
        mu = valid_group_x.mean()
        std = valid_group_x.std(unbiased=False) + 1e-8
        group_out = out[s:e]
        group_out[group_mask] = (group_x[group_mask] - mu) / std
        out[s:e] = group_out
        out_valid_mask[s:e] = group_mask
    return out, out_valid_mask

def train_one_step(
    args,
    device,
    transformer,
    vae,
    # reward_model,
    # tokenizer,
    optimizer,
    lr_scheduler,
    prompt_embeds, 
    prompt_attention_masks, 
    prompt, 
    original_length,
    image_latents,
    batch_examples,
    noise_scheduler,
    max_grad_norm,
    # preprocess_val,
    ema_handler,
    lpips_loss=None,
    ssim_loss=None,
    do_visualize: bool = False,
):
    train_one_step_start = time.perf_counter()
    total_loss = 0.0
    step_metrics: Dict[str, float] = {}
        
    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(prompt_embeds)
        prompt_attention_masks = repeat_tensor(prompt_attention_masks)
        image_latents = repeat_tensor(image_latents)
        original_length = repeat_tensor(original_length)
        batch_examples = [example for example in batch_examples for _ in range(args.num_generations)]
        
        if isinstance(prompt, str):
            prompt = [prompt] * args.num_generations
        elif isinstance(prompt, list):
            prompt = [item for item in prompt for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
    else:
        encoder_hidden_states = prompt_embeds
        batch_examples = list(batch_examples)

    sample_start = time.perf_counter()
    (
        rewards_dict,
        reward_valid_masks,
        all_latents,
        all_log_probs,
        sigma_schedule,
        all_txt_seq_lens,
        all_image_latents,
        all_img_shapes,
        all_num_output_frames,
        vis_data,
    ) = sample_reference_model(
            args,
            device, 
            transformer,
            vae,
            encoder_hidden_states, 
            prompt_attention_masks, 
            original_length,
            image_latents,
            batch_examples,
            # reward_model,
            # tokenizer,
            prompt,
            # preprocess_val,
            lpips_loss,
            ssim_loss,
            do_visualize=do_visualize,
        )
    step_metrics["train-time/sample_reference"] = time.perf_counter() - sample_start
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        # "rewards": reward.to(torch.float32),
        "txt_seq_lens": all_txt_seq_lens,
        "encoder_hidden_states": encoder_hidden_states,
        "prompt_attention_masks": prompt_attention_masks,
        "image_latents": all_image_latents,
        "original_length": original_length,
        "num_output_frames": torch.tensor(all_num_output_frames, device=device, dtype=torch.long),
    }
    
    # raw rewards from new terms
    samples["r1_layer_blend_ssim_raw"] = rewards_dict["r1_layer_blend_ssim"].detach().to(torch.float32)
    samples["r1_layer_blend_psnr_raw"] = rewards_dict["r1_layer_blend_psnr"].detach().to(torch.float32)
    samples["r2_comp_ssim_raw"] = rewards_dict["r2_comp_ssim"].detach().to(torch.float32)
    samples["r2_comp_psnr_raw"] = rewards_dict["r2_comp_psnr"].detach().to(torch.float32)
    samples["r2_comp_lpips_raw"] = rewards_dict["r2_comp_lpips"].detach().to(torch.float32)
    samples["r3_bg_blend_lpips_raw"] = rewards_dict["r3_bg_blend_lpips"].detach().to(torch.float32)
    samples["r4_fg_boundary_alpha_l1_raw"] = rewards_dict["r4_fg_boundary_alpha_l1"].detach().to(torch.float32)
    samples["r_final_raw"]  = rewards_dict["final"].detach().to(torch.float32)
    samples["r1_layer_blend_ssim_valid"] = reward_valid_masks["r1_layer_blend_ssim"]
    samples["r1_layer_blend_psnr_valid"] = reward_valid_masks["r1_layer_blend_psnr"]
    samples["r2_comp_ssim_valid"] = reward_valid_masks["r2_comp_ssim"]
    samples["r2_comp_psnr_valid"] = reward_valid_masks["r2_comp_psnr"]
    samples["r2_comp_lpips_valid"] = reward_valid_masks["r2_comp_lpips"]
    samples["r3_bg_blend_lpips_valid"] = reward_valid_masks["r3_bg_blend_lpips"]
    samples["r4_fg_boundary_alpha_l1_valid"] = reward_valid_masks["r4_fg_boundary_alpha_l1"]
    samples["rewards"] = samples["r_final_raw"]

    advantage_start = time.perf_counter()
    # ======== term-wise group z-score ========
    G = args.num_generations if args.use_group else 1
    term_specs = [
        ("r1_layer_blend_ssim_raw", "r1_layer_blend_ssim_valid", args.use_r1_layer_blend_ssim, float(args.w_r1_layer_blend_ssim)),
        ("r1_layer_blend_psnr_raw", "r1_layer_blend_psnr_valid", args.use_r1_layer_blend_psnr, float(args.w_r1_layer_blend_psnr)),
        ("r2_comp_ssim_raw", "r2_comp_ssim_valid", args.use_r2_comp_ssim, float(args.w_r2_comp_ssim)),
        ("r2_comp_psnr_raw", "r2_comp_psnr_valid", args.use_r2_comp_psnr, float(args.w_r2_comp_psnr)),
        ("r2_comp_lpips_raw", "r2_comp_lpips_valid", args.use_r2_comp_lpips, float(args.w_r2_comp_lpips)),
        ("r3_bg_blend_lpips_raw", "r3_bg_blend_lpips_valid", args.use_r3_bg_blend_lpips, float(args.w_r3_bg_blend_lpips)),
        ("r4_fg_boundary_alpha_l1_raw", "r4_fg_boundary_alpha_l1_valid", args.use_r4_fg_boundary_alpha_l1, float(args.w_r4_fg_boundary_alpha_l1)),
    ]

    z_numerator = torch.zeros_like(samples["r_final_raw"], dtype=torch.float32)
    z_denominator = torch.zeros_like(samples["r_final_raw"], dtype=torch.float32)
    for raw_name, valid_name, enabled, weight in term_specs:
        if enabled and weight > 0:
            z_term, z_valid = masked_zscore_by_group(
                samples[raw_name],
                samples[valid_name],
                G,
            )
            z_numerator = z_numerator + weight * z_term
            z_denominator = z_denominator + weight * z_valid.to(torch.float32)
    if torch.all(z_denominator <= 0):
        raise ValueError("No reward terms enabled to form final advantage.")
    if torch.any(z_denominator <= 0):
        raise ValueError("Some samples have no valid reward terms to form final advantage.")
    z_final = z_numerator / z_denominator.clamp_min(1e-8)

    # final group z-score -> advantage
    advantages, _ = masked_zscore_by_group(
        z_final,
        torch.ones_like(z_final, dtype=torch.bool),
        G,
    )
    samples["advantages"] = advantages
    step_metrics["train-time/advantage"] = time.perf_counter() - advantage_start

    reward_stats_start = time.perf_counter()
    reward_metric_specs = [
        ("train-reward/r1_layer_blend_ssim", samples["r1_layer_blend_ssim_raw"], samples["r1_layer_blend_ssim_valid"], args.use_r1_layer_blend_ssim, float(args.w_r1_layer_blend_ssim)),
        ("train-reward/r1_layer_blend_psnr", samples["r1_layer_blend_psnr_raw"], samples["r1_layer_blend_psnr_valid"], args.use_r1_layer_blend_psnr, float(args.w_r1_layer_blend_psnr)),
        ("train-reward/r2_comp_ssim", samples["r2_comp_ssim_raw"], samples["r2_comp_ssim_valid"], args.use_r2_comp_ssim, float(args.w_r2_comp_ssim)),
        ("train-reward/r2_comp_psnr", samples["r2_comp_psnr_raw"], samples["r2_comp_psnr_valid"], args.use_r2_comp_psnr, float(args.w_r2_comp_psnr)),
        ("train-reward/r2_comp_lpips", samples["r2_comp_lpips_raw"], samples["r2_comp_lpips_valid"], args.use_r2_comp_lpips, float(args.w_r2_comp_lpips)),
        ("train-reward/r3_bg_blend_lpips", samples["r3_bg_blend_lpips_raw"], samples["r3_bg_blend_lpips_valid"], args.use_r3_bg_blend_lpips, float(args.w_r3_bg_blend_lpips)),
        ("train-reward/r4_fg_boundary_alpha_l1", samples["r4_fg_boundary_alpha_l1_raw"], samples["r4_fg_boundary_alpha_l1_valid"], args.use_r4_fg_boundary_alpha_l1, float(args.w_r4_fg_boundary_alpha_l1)),
    ]
    for prefix, tensor, valid_mask, enabled, weight in reward_metric_specs:
        if enabled and weight > 0:
            add_masked_tensor_stats(step_metrics, prefix, tensor, valid_mask)
    add_tensor_stats(step_metrics, "train-reward/final", gather_tensor(samples["r_final_raw"]))
    add_tensor_stats(step_metrics, "train-advantage/z_final", gather_tensor(z_final.detach().to(torch.float32)))
    add_tensor_stats(step_metrics, "train-advantage/value", gather_tensor(samples["advantages"].detach().to(torch.float32)))
    step_metrics["train-time/reward_stats"] = time.perf_counter() - reward_stats_start
    
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ]
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)
    if train_timesteps <= 0:
        raise ValueError("timestep_fraction is too small; no policy update steps would run.")
    ratio_values = []
    policy_update_start = time.perf_counter()
    for i,sample in list(enumerate(samples_batched_list)):
        shapes = all_img_shapes[i]   

        frames = int(sample["num_output_frames"].item())
        F_cond = len(shapes) - frames

        img_shapes = [[
            *shapes[:frames],    # 输出帧
            *shapes[frames:],    # 条件帧
        ]]
        
        for _ in range(train_timesteps):
            clip_range = args.clip_range
            adv_clip_max = args.adv_clip_max
            sample_original_length = int(sample["original_length"].item())
             
            # _, z_dim, F_cond, H_cond, W_cond = sample["image_latents"].shape
            # latent_h = H_cond
            # latent_w = W_cond

            # img_shapes = [[
            # *[(1, latent_h // 2, latent_w // 2) for _ in range(frames)],   # 输出帧
            # *[(1, H_cond   // 2, W_cond   // 2) for _ in range(F_cond)],   # 条件帧
            # ] for _ in range(sample["image_latents"].shape[0])]  # batch_size
            
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"][:, :sample_original_length],
                sample["prompt_attention_masks"][:, :sample_original_length],
                sample["txt_seq_lens"],
                # [[
                #     (1, args.h // 8 // 2, args.w // 8// 2),
                #     (1, calculated_height[0].item() // 8 // 2, calculated_width[0].item() // 8 // 2),
                # ]],
                img_shapes,
                sample["image_latents"],
                transformer,
                sample["timesteps"][:,_],
                perms[i][_],
                sigma_schedule,
            )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])
            ratio_values.append(ratio.detach().to(torch.float32).reshape(-1))

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        dist.barrier()
    if ratio_values:
        add_tensor_stats(step_metrics, "train-policy/ratio", gather_tensor(torch.cat(ratio_values, dim=0)))
    step_metrics["train-time/policy_update"] = time.perf_counter() - policy_update_start

    grad_norm_value = grad_norm.detach().to(torch.float32)
    if dist.is_initialized():
        dist.all_reduce(grad_norm_value, op=dist.ReduceOp.AVG)

    step_metrics["train-time/train_one_step"] = time.perf_counter() - train_one_step_start
    return total_loss, grad_norm_value.item(), vis_data, step_metrics


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    tb_writer = None
    if rank == 0:
        tb_log_dir = args.logging_dir
        if args.output_dir is not None and not os.path.isabs(tb_log_dir):
            tb_log_dir = os.path.join(args.output_dir, tb_log_dir)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        main_print(f"--> TensorBoard logging to {tb_log_dir}")

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    
    # --- reward metrics ---
    lpips_loss = None
    ssim_loss = None

    need_lpips = args.use_r2_comp_lpips or args.use_r3_bg_blend_lpips
    need_ssim = args.use_r1_layer_blend_ssim or args.use_r2_comp_ssim

    if need_lpips:
        import lpips
        lpips_loss = lpips.LPIPS(net='vgg').to(device)
        lpips_loss.eval()
        for p in lpips_loss.parameters():
            p.requires_grad_(False)

    if need_ssim:
        import piq
        ssim_loss = piq.SSIMLoss(data_range=1.0).to(device)
        
    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    
    # from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, QwenImageTransformerBlock
    # transformer = QwenImageTransformer2DModel.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         subfolder="transformer",
    #         torch_dtype = torch.float32
    # )
    
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

    # 加载自定义 pipeline 
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            #revision=args.revision, 
            #torch_dtype=torch.float32,  
        )
 
    modules = QwenImageEditModules.from_pipeline(pipeline)
        
    if args.vae_model_path is not None:
        modules.vae = load_vae_from_local_dir(args.vae_model_path) 
        
    pipeline = QwenImageEditPlusPipeline(**modules.to_dict())
    
    # TODO: remove after debug
    for param in pipeline.transformer.parameters():
        assert param.requires_grad == True
        
    modules.requires_grad_(False)
    
    # TODO: remove after debug
    for param in pipeline.transformer.parameters():
        assert param.requires_grad == False
    
    #cast all non-trainable weights (vae, text_encoder and transformer) to half-precision?
    
    if args.load_lora:
        pipeline.load_lora_weights(args.load_lora, "default")
        main_print(f"--> Loaded LoRA weights from {args.load_lora}")
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
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        modules.transformer.add_adapter(transformer_lora_config)
        main_print("--> LoRA layers added to transformer") 
    
    modules.text_encoder.to(device=device, dtype=torch.bfloat16)
    vae = modules.vae.to(device=device, dtype=torch.bfloat16)
    transformer = modules.transformer
    
    # Setup FSDP configuration
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,  
        num_replicate=1,
        num_shard=world_size,
        mixed_precision_dtype=torch.bfloat16,
        use_device_mesh=False, 
    )
    #只包裹transformer
    transformer = fsdp_wrapper(transformer, fsdp_config,)

    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    apply_fsdp_checkpointing(
            transformer, (QwenImageTransformerBlock), args.selective_checkpointing
        )


    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_strategy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")
    
    if rank == 0:
        trainable = [n for n, p in transformer.named_parameters() if p.requires_grad]
        frozen = [n for n, p in transformer.named_parameters() if not p.requires_grad]

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000, 
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    if args.train_batch_size != 1:
        raise ValueError("Raw GRPO data loading currently supports train_batch_size=1 only.")

    dataset_manager = DatasetManager(args.data_json_path).set_default_class(
        partial(TI2IDataset, multiple_of=32, max_pixels=args.max_pixels)
    )
    train_dataset = dataset_manager.get_split(args.dataset_split)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )
    

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=raw_ti2i_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    
    
    #vae.enable_tiling()

    # if rank <= 0:
    #     project = "qwenimage_edit"
    #     wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )

    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, 100000), 
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )



    step_times = deque(maxlen=100)
    global_step = init_steps

    # The number of epochs 1 is a random value; you can also set the number of epochs to be two.
    for epoch in range(3):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            prepare_batch_start = time.perf_counter()
            prompt_embeds, prompt_attention_masks, image_latents, prompt, original_length, batch_examples = prepare_online_grpo_batch(
                batch,
                pipeline,
                device,
                vae.dtype,
            )
            prepare_batch_time = time.perf_counter() - prepare_batch_start
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_masks = prompt_attention_masks.to(device)
            image_latents = image_latents.to(device)
            # if (step-1) % args.checkpointing_steps == 0 and step!=1:
            #     save_checkpoint(transformer, rank, args.output_dir,
            #                     step, epoch)
            #     if args.use_ema:
            #         save_ema_checkpoint(ema_handler, rank, args.output_dir, step, epoch, dict(transformer.config))


            #     dist.barrier()
            
            #lora权重保存
            if (step - 1) % args.checkpointing_steps == 0 and step != 1:
                if args.save_only_lora:
                    from torch.distributed.fsdp import (
                        StateDictType,
                        FullStateDictConfig,
                        FullyShardedDataParallel as FSDP,
                    )
                    # 1. 从 FSDP shard 里拿到 full state dict 到 CPU
                    with FSDP.state_dict_type(
                        transformer,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(
                            offload_to_cpu=True,
                            rank0_only=True,
                        ),
                    ):
                        full_state_dict = transformer.state_dict()
                        
                    if rank == 0:
                        # 2. 提取 LoRA 对应的权重
                        transformer_lora_layers = get_peft_model_state_dict(
                            model=transformer,
                            state_dict=full_state_dict,
                        )
                        # 可选：32 or bf16
                        for k, v in transformer_lora_layers.items():
                            transformer_lora_layers[k] = v.to(torch.bfloat16) 
                         
                        # 3. 交给 pipeline 去存 LoRA 权重
                        save_dir = os.path.join(
                            args.output_dir, f"lora-only-{global_step:06d}-{epoch}-{step}"
                        )
                        os.makedirs(save_dir, exist_ok=True)

                        main_print(f"--> saving LoRA-only checkpoint at {save_dir}")
                        pipeline.save_lora_weights(
                            save_directory=save_dir,
                            transformer_lora_layers=transformer_lora_layers,
                            is_main_process=True,
                        )
                else:
                    save_checkpoint(
                        transformer,
                        rank,
                        args.output_dir,
                        step,
                        epoch,
                    )
                    if args.use_ema:
                        save_ema_checkpoint(
                            ema_handler,
                            rank,
                            args.output_dir,
                            step,
                            epoch,
                            dict(transformer.config),
                        )

                dist.barrier()
            
            if step > (args.max_train_steps+1):
                break
            
            # ==== 这里决定这一 step 是否做可视化 ====
            do_visualize = (
                (args.vis_interval is not None)
                and (args.vis_interval > 0)
                and (global_step % args.vis_interval == 0)
            )
            
            loss, grad_norm, vis_data, step_metrics = train_one_step(
                args,
                device, 
                transformer,
                vae,
                # reward_model,
                # processor,
                optimizer,
                lr_scheduler,
                prompt_embeds, 
                prompt_attention_masks, 
                prompt, 
                original_length,
                image_latents,
                batch_examples,
                noise_scheduler,
                args.max_grad_norm,
                # preprocess_val,
                ema_handler,
                lpips_loss=lpips_loss,
                ssim_loss=ssim_loss,
                do_visualize=do_visualize
            )

            ema_update_time = 0.0
            if args.use_ema and ema_handler:
                ema_update_start = time.perf_counter()
                ema_handler.update(transformer)
                ema_update_time = time.perf_counter() - ema_update_start
    
            # 如果当前步做了可视化，就 gather 所有 rank 的样本并统一存图/打板
            vis_gather_time = 0.0
            if do_visualize:
                vis_gather_start = time.perf_counter()
                gathered_vis_data = all_gather_flattened_objects([] if vis_data is None else [vis_data])
                vis_gather_time = time.perf_counter() - vis_gather_start
            else:
                gathered_vis_data = []

            vis_save_time = 0.0
            vis_tb_time = 0.0
            next_global_step = global_step + 1
            if rank == 0 and gathered_vis_data:
                vis_dir = args.vis_dir or os.path.join(args.output_dir, "train_vis")
                vis_save_start = time.perf_counter()
                save_vis_single_images(
                    save_dir=vis_dir,
                    global_step=next_global_step,
                    vis_data_list=gathered_vis_data,
                )
                grid_image = save_vis_grid(
                    save_dir=vis_dir,
                    global_step=next_global_step,
                    vis_data_list=gathered_vis_data,
                )
                vis_save_time = time.perf_counter() - vis_save_start
                vis_tb_start = time.perf_counter()
                log_vis_to_tensorboard(
                    writer=tb_writer,
                    global_step=next_global_step,
                    vis_data_list=gathered_vis_data,
                    grid_image=grid_image,
                )
                vis_tb_time = time.perf_counter() - vis_tb_start

            step_total_time = time.time() - start_time
            step_times.append(step_total_time)
            avg_step_time = sum(step_times) / len(step_times)

            global_step = next_global_step
            scalar_logs = {
                "train-policy/loss": loss,
                "train-policy/lr": lr_scheduler.get_last_lr()[0],
                "train-policy/grad_norm": grad_norm,
                "train-time/prepare_batch": prepare_batch_time,
                "train-time/ema_update": ema_update_time,
                "train-time/vis_gather": vis_gather_time,
                "train-time/vis_save": vis_save_time,
                "train-time/vis_tensorboard": vis_tb_time,
                "train-time/step_total": step_total_time,
                "train-time/step_total_avg": avg_step_time,
            }
            scalar_logs.update(step_metrics)
            if rank == 0:
                add_scalar_dict(tb_writer, global_step, scalar_logs)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "reward": f"{step_metrics.get('train-reward/final/mean', 0.0):.4f}",
                    "step": f"{step_total_time:.2f}s",
                    "sample": f"{step_metrics.get('train-time/sample_reference', 0.0):.2f}s",
                    "policy": f"{step_metrics.get('train-time/policy_update', 0.0):.2f}s",
                }
            )
            progress_bar.update(1)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument(
        "--data_json_path",
        type=str,
        required=True,
        help="DatasetManager config path (json/jsonc) for raw TI2I training data.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Split name inside data_json_path to load.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help="Max total pixels for one sample before joint downscaling, matching the old preprocess behavior.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_strategy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--loss_coef",
        type=float,
        default=1.0,   
        help="the global loss should be divided by",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=True,
        help="whether compute advantages for each prompt",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    
    parser.add_argument(
        "--ignore_last",
        action="store_true",
        default=False,
        help="whether ignore last step of mdp",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep downsample ratio",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Enable Exponential Moving Average of model weights."
    )
    
    # lora
    parser.add_argument(
        "--load_lora",
        type=str,
        default=None,
        help="Path to existing LoRA weights to load.",
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help="Comma separated list of target modules for LoRA. If None, use default list.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=256,
        help="LoRA rank (dimension of A/B).",
    )
    parser.add_argument(
        "--save_only_lora",
        action="store_true",
        help="If set, only save LoRA weights instead of full transformer checkpoints.",
    )

    # new rewards
    parser.add_argument("--use_r1_layer_blend_ssim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_r1_layer_blend_ssim", type=float, default=1.0)
    parser.add_argument("--use_r1_layer_blend_psnr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_r1_layer_blend_psnr", type=float, default=1.0)

    parser.add_argument("--use_r2_comp_ssim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_r2_comp_ssim", type=float, default=1.0)
    parser.add_argument("--use_r2_comp_psnr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_r2_comp_psnr", type=float, default=1.0)
    parser.add_argument("--use_r2_comp_lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_r2_comp_lpips", type=float, default=1.0)

    parser.add_argument("--use_r3_bg_blend_lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_r3_bg_blend_lpips", type=float, default=1.0)

    parser.add_argument("--use_r4_fg_boundary_alpha_l1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_r4_fg_boundary_alpha_l1", type=float, default=1.0)

    parser.add_argument("--boundary_alpha_low", type=float, default=0.05)
    parser.add_argument("--boundary_alpha_high", type=float, default=0.95)
    parser.add_argument("--boundary_blur_px", type=int, default=10)
    parser.add_argument("--psnr_max_db", type=float, default=50.0)

    # visualization during training
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=0,
        help="If > 0, save a visualization grid every N training steps (global step).",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="Directory to save training visualization grids. If None, will use <output_dir>/train_vis.",
    )


    args = parser.parse_args()
    main(args)
