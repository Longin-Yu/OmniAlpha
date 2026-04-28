import argparse  
import os
import json
import re
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate.logging import get_logger
from tqdm import tqdm
from PIL import Image

from alpha.pipelines.qwen_image_edit import (
    CustomQwenImageEditPlusPipeline as QwenImageEditPlusPipeline,
    QwenImageEditModules,
)
from alpha.vae.modeling import load_vae_from_local_dir

logger = get_logger(__name__)


def center_crop_to_multiple(image: Image.Image, multiple: int = 32):
    w, h = image.size  

    if w < multiple or h < multiple:
        raise ValueError(f"Image too small ({w}x{h}) for multiple={multiple}")

    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    # 如果刚好已经是合格尺寸，就不用 crop 了
    if new_w == w and new_h == h:
        return image, h, w

    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    right  = left + new_w
    bottom = top + new_h

    cropped = image.crop((left, top, right, bottom))
    return cropped, new_h, new_w


def get_pixel_constraint_ratio_from_paths(image_paths, max_pixels=None):
    if max_pixels is None:
        return 1.0
    total_pixels = 0
    for path in image_paths:
        with Image.open(path) as img:
            total_pixels += img.width * img.height
    if total_pixels <= max_pixels:
        return 1.0
    return math.sqrt(max_pixels / total_pixels)


class T5dataset(Dataset):
    def __init__(self, jsonl_path, base_image_path):
        self.jsonl_path = jsonl_path
        self.base_image_path = base_image_path
        
        self.data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    continue
                
    def _build_sample_filename(self, item, input_image_rel):
        # Old format may provide explicit id.
        if "id" in item:
            return str(item["id"])

        # New format has no id; derive a collision-resistant key from relative path.
        rel_no_ext = os.path.splitext(input_image_rel)[0]
        safe_name = rel_no_ext.replace("\\", "__").replace("/", "__")
        return safe_name if safe_name else "sample"

    def __getitem__(self, idx):
        item = self.data[idx]
        input_images = item["input_images"]
        output_images = item.get("output_images", [])
        resolved_input_images = [
            p if (p and os.path.isabs(p)) else os.path.join(self.base_image_path, p)
            for p in input_images
        ]
        resolved_output_images = [
            p if (p and os.path.isabs(p)) else os.path.join(self.base_image_path, p)
            for p in output_images
        ]
        prompt = item["prompt"]
        return {
            "input_images": resolved_input_images,
            "pixel_constraint_images": resolved_input_images + resolved_output_images,
            "prompt": prompt,
            "filename": self._build_sample_filename(item, input_images[0] if input_images else "")
        }

    def __len__(self):
        return len(self.data)

# train_batch_size=1
def calculate_embeds(pipe, data, device, max_pixels):
    # 1. 读图（RGBA 以配合 4 通道 VAE）
    input_image_paths = [x[0] for x in data["input_images"]]
    images = [Image.open(p).convert("RGBA") for p in input_image_paths]
    image = images[0]
    
    # print("input_image:", data["input_image"], flush=True)
    # print("pixel_constraint_images raw:", data["pixel_constraint_images"], flush=True)
    # print("type:", type(data["pixel_constraint_images"]), flush=True)
    # print("first elem:", data["pixel_constraint_images"][0], flush=True)
    
    pixel_constraint_paths = [x[0] for x in data["pixel_constraint_images"]]
    # print("pixel_constraint_paths:", pixel_constraint_paths, flush=True)

    ratio = get_pixel_constraint_ratio_from_paths(
        pixel_constraint_paths, max_pixels=max_pixels
    )
    
    if ratio < 1.0:
        new_w = max(1, int(image.width * ratio))
        new_h = max(1, int(image.height * ratio))
        images = [
            img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            for img in images
        ]
        image = images[0]

    cropped_images = []
    for img in images:
        cropped_img, cropped_h, cropped_w = center_crop_to_multiple(
            img, multiple=32
        )
        cropped_images.append(cropped_img)

    cropped_image = cropped_images[0]
    # print(f"{cropped_h=}, {cropped_w=}", flush=True)    
    
    # 2. 准备给 text encoder 用的条件图
    cond_images, _ = pipe.prepare_images( 
        cropped_images,
        "condition",
        reshape=False,
    )

    with torch.no_grad():
        prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            image=cond_images,
            prompt=[data['prompt'][0]],
            device=device,
        )

    # 3. 用自定义 VAE 编码 image -> image_latents
    vae_images, _ = pipe.prepare_images(
        cropped_images,
        "vae",
        reshape=False,
    )  

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        image_latents_list = [
            pipe._encode_vae_image(img.to(device=device, dtype=torch.bfloat16), generator=None)
            for img in vae_images
        ]
        image_latents = torch.cat(image_latents_list, dim=2)   # (1, z_dim, F_cond, H, W)
    
    # print("image_latents.shape =", image_latents.shape, flush=True)
    
    return prompt_embeds, prompt_attention_mask, image_latents, cropped_h, cropped_w, ratio


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=local_rank
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_attention_mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "image_latents"), exist_ok=True)

    # 数据集
    train_dataset = T5dataset(
        jsonl_path=args.prompt_dir,
        base_image_path=args.base_image_path
    )
    
    sampler = DistributedSampler(
        train_dataset, rank=local_rank, num_replicas=world_size, shuffle=False
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # 加载自定义 QwenImageEditPlusPipeline + 自定义 VAE
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
    )
    modules = QwenImageEditModules.from_pipeline(pipeline)

    if args.vae_model_path is not None:
        modules.vae = load_vae_from_local_dir(args.vae_model_path)
        print(f"[preprocess] loaded custom VAE from {args.vae_model_path}")

    pipeline = QwenImageEditPlusPipeline(**modules.to_dict()).to(
        device=device, dtype=torch.bfloat16
    )
    print("QwenImageEditPlus pipeline with custom VAE loaded")

    json_data = []

    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode(), torch.autocast("cuda"):
            prompt_embeds, prompt_attention_mask, image_latents, cropped_h, cropped_w, resize_ratio = \
                calculate_embeds(pipeline, data, device, args.max_pixels)

            # === 序列 padding 逻辑 ===
            original_length = prompt_embeds.shape[1]
            # target_length = 5000
            target_length = 1500
            pad_len = target_length - original_length

            prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, pad_len), "constant", 0)
            prompt_attention_mask = F.pad(prompt_attention_mask, (0, pad_len), "constant", 0)

            if args.vae_debug:
                latents = data.get("latents", None)  # 兼容旧逻辑，不用也没关系

            for idx, video_name in enumerate(data["filename"]):
                prompt_embed_path = os.path.join(
                    args.output_dir, "prompt_embed", video_name + ".pt"
                )
                prompt_attention_mask_path = os.path.join(
                    args.output_dir, "prompt_attention_mask", video_name + ".pt"
                )
                image_latents_path = os.path.join(
                    args.output_dir, "image_latents", video_name + ".pt"
                )

                torch.save(prompt_embeds[idx], prompt_embed_path)
                torch.save(prompt_attention_mask[idx], prompt_attention_mask_path)
                torch.save(image_latents[idx], image_latents_path)

                item = {}
                item["prompt_embed_path"] = video_name + ".pt"
                item["prompt_attention_mask"] = video_name + ".pt"
                item["image_latents"] = video_name + ".pt"
                item["prompt"] = data["prompt"][idx]
                item["original_length"] = original_length  
                item["resize_ratio"] = float(resize_ratio)
                item["calculated_height"] = int(cropped_h)  
                item["calculated_width"] = int(cropped_w)   
                
                json_data.append(item)

    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "preprocess.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vae_debug", 
        action="store_true"
    )
    parser.add_argument(
        "--prompt_dir", 
        type=str, 
        default="./empty.txt"
    )
    parser.add_argument(
    "--base_image_path",
    type=str,
    default="./data/SEED-Data-Edit-Part2-3/real_editing/images",
    help="JSONL 里 input_image 的相对路径所对应的根目录",
    )

    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help="Max total pixels for preprocessing images. Images are uniformly downscaled if exceeded.",
    )

    args = parser.parse_args()
    main(args)