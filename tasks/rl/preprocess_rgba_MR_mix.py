import argparse
import json
import math
import os
import random
import re
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from alpha.pipelines.qwen_image_edit import (
    CustomQwenImageEditPlusPipeline as QwenImageEditPlusPipeline,
    QwenImageEditModules,
)
from alpha.vae.modeling import load_vae_from_local_dir


def center_crop_to_multiple(image: Image.Image, multiple: int = 32):
    w, h = image.size

    if w < multiple or h < multiple:
        raise ValueError(f"Image too small ({w}x{h}) for multiple={multiple}")

    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    if new_w == w and new_h == h:
        return image, h, w

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
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


def load_jsonl_file(path: str):
    if not str(path).lower().endswith(".jsonl"):
        raise ValueError(f"Only .jsonl is supported for dataset files, got: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: invalid jsonl at line {line_idx}: {e}") from e
    return rows


def load_json_or_jsonc_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Remove // comments and /* ... */ comments for simple jsonc compatibility.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    return json.loads(text)


class T5dataset(Dataset):
    def __init__(self, jsonl_path, base_image_path, source_name="default"):
        self.jsonl_path = jsonl_path
        self.base_image_path = base_image_path
        self.source_name = source_name

        data = load_jsonl_file(self.jsonl_path)
        if not isinstance(data, list):
            raise ValueError(f"{self.jsonl_path} must contain a list-like JSON/JSONL payload.")
        self.data = data

    def _build_sample_filename(self, item, input_image_rel):
        if "id" in item:
            return str(item["id"])
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
            "filename": self._build_sample_filename(item, input_images[0] if input_images else ""),
            "source_dataset": self.source_name,
        }

    def __len__(self):
        return len(self.data)


class MultiSourceT5Dataset(Dataset):
    """
    Config schema (json/jsonc):
    {
      "datasets": {
        "dataset_name": {
          "jsonl_path": ".../data.jsonl",     // also accepts key "data_path"
          "base_image_path": ".../images"     // optional; falls back to default_base_image_path
        }
      },
      "splits": {
        "train": [
          {
            "dataset": "dataset_name",
            "starts": 0,                      // optional, supports negative index
            "ends": 1000,                     // optional, supports negative index
            "repeat": 1,                      // optional
            "sample": {                       // optional
              "count": 500,                   // optional
              "ratio": 0.2,                   // optional
              "seed": 42                      // optional
            },
            "weight": 1.0                     // optional, post-slice resample factor
          }
        ]
      }
    }
    """

    def __init__(
        self,
        config_path,
        split_name="train",
        default_base_image_path="",
        global_seed=0,
        strict_unique_filename=True,
    ):
        self.config_path = config_path
        self.split_name = split_name
        self.default_base_image_path = default_base_image_path
        self.global_seed = int(global_seed)
        self.strict_unique_filename = strict_unique_filename

        self.config = load_json_or_jsonc_config(config_path)
        if "datasets" not in self.config or "splits" not in self.config:
            raise ValueError("dataset config must contain both top-level keys: datasets, splits")
        if split_name not in self.config["splits"]:
            raise ValueError(f"split '{split_name}' not found in dataset config")

        self._dataset_cache = {}
        self.data = []
        self._build_split_records()

    @staticmethod
    def _normalize_slice_bounds(starts, ends, n):
        s = 0 if starts is None else int(starts)
        e = n if ends is None else int(ends)
        while s < 0:
            s += n
        while e < 0:
            e += n
        s = max(0, min(s, n))
        e = max(0, min(e, n))
        if e < s:
            e = s
        return s, e

    @staticmethod
    def _build_sample_filename(item, input_image_rel):
        if "id" in item:
            return str(item["id"])
        rel_no_ext = os.path.splitext(input_image_rel)[0]
        safe_name = rel_no_ext.replace("\\", "__").replace("/", "__")
        return safe_name if safe_name else "sample"

    def _get_dataset_records(self, name):
        if name in self._dataset_cache:
            return self._dataset_cache[name]
        ds_cfg = deepcopy(self.config["datasets"].get(name))
        if ds_cfg is None:
            raise ValueError(f"dataset '{name}' not found in config.datasets")

        jsonl_path = ds_cfg.get("jsonl_path", ds_cfg.get("data_path"))
        if not jsonl_path:
            raise ValueError(f"dataset '{name}' requires jsonl_path (or data_path)")
        base_image_path = ds_cfg.get("base_image_path", self.default_base_image_path)

        records = load_jsonl_file(jsonl_path)
        if not isinstance(records, list):
            raise ValueError(f"{jsonl_path} must contain a list-like JSON/JSONL payload.")
        self._dataset_cache[name] = (records, base_image_path)
        return self._dataset_cache[name]

    @staticmethod
    def _sample_records(records, sample_cfg, fallback_seed):
        if not sample_cfg:
            return records
        sample_cfg = deepcopy(sample_cfg)
        count = len(records)
        if "ratio" in sample_cfg:
            ratio = float(sample_cfg["ratio"])
            ratio = max(0.0, ratio)
            count = min(count, int(len(records) * ratio))
        elif "count" in sample_cfg:
            count = min(len(records), max(0, int(sample_cfg["count"])))
        seed = int(sample_cfg.get("seed", fallback_seed))
        rng = random.Random(seed)
        if count >= len(records):
            return records
        if count <= 0:
            return []
        idx = rng.sample(range(len(records)), count)
        return [records[i] for i in idx]

    @staticmethod
    def _apply_weight(records, weight, seed):
        weight = float(weight)
        if weight <= 0 or len(records) == 0:
            return []
        if abs(weight - 1.0) < 1e-8:
            return records

        target_n = int(round(len(records) * weight))
        if target_n <= 0:
            return []

        rng = random.Random(int(seed))
        if target_n <= len(records):
            idx = rng.sample(range(len(records)), target_n)
            return [records[i] for i in idx]
        idx = rng.choices(range(len(records)), k=target_n)
        return [records[i] for i in idx]

    def _build_split_records(self):
        split_entries = self.config["splits"][self.split_name]
        filename_seen = {}
        merged = []

        for entry_id, entry in enumerate(split_entries):
            ds_name = entry.get("dataset")
            if not ds_name:
                raise ValueError(f"entry #{entry_id} in split '{self.split_name}' is missing dataset")

            records, base_image_path = self._get_dataset_records(ds_name)
            n = len(records)
            s, e = self._normalize_slice_bounds(entry.get("starts"), entry.get("ends"), n)
            selected = records[s:e]

            repeat = int(entry.get("repeat", 1))
            if repeat < 1:
                repeat = 1
            if repeat > 1:
                selected = selected * repeat

            sample_seed = self.global_seed + entry_id * 10007
            selected = self._sample_records(selected, entry.get("sample"), sample_seed)
            selected = self._apply_weight(selected, entry.get("weight", 1.0), sample_seed + 17)

            for rec in selected:
                input_images = rec.get("input_images", [])
                output_images = rec.get("output_images", [])
                prompt = rec.get("prompt")
                if not isinstance(input_images, list) or len(input_images) == 0:
                    raise ValueError(f"dataset={ds_name}: invalid input_images in record: {rec}")
                if not isinstance(output_images, list):
                    raise ValueError(f"dataset={ds_name}: invalid output_images in record: {rec}")
                if not isinstance(prompt, str):
                    raise ValueError(f"dataset={ds_name}: invalid prompt in record: {rec}")

                resolved_input_images = [
                    p if (p and os.path.isabs(p)) else os.path.join(base_image_path, p)
                    for p in input_images
                ]
                resolved_output_images = [
                    p if (p and os.path.isabs(p)) else os.path.join(base_image_path, p)
                    for p in output_images
                ]

                filename = self._build_sample_filename(rec, input_images[0] if input_images else "")
                if self.strict_unique_filename and filename in filename_seen:
                    old_src = filename_seen[filename]
                    raise ValueError(
                        f"duplicate sample filename detected: '{filename}'. "
                        f"current dataset={ds_name}, previous dataset={old_src}. "
                        "Please resolve duplicate input_images[0] keys across mixed datasets."
                    )
                filename_seen[filename] = ds_name

                merged.append(
                    {
                        "input_images": resolved_input_images,
                        "pixel_constraint_images": resolved_input_images + resolved_output_images,
                        "prompt": prompt,
                        "filename": filename,
                        "source_dataset": ds_name,
                    }
                )

        self.data = merged

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def calculate_embeds(pipe, data, device, max_pixels):
    input_image_paths = [x[0] for x in data["input_images"]]
    images = [Image.open(p).convert("RGBA") for p in input_image_paths]
    image = images[0]

    pixel_constraint_paths = [x[0] for x in data["pixel_constraint_images"]]
    ratio = get_pixel_constraint_ratio_from_paths(pixel_constraint_paths, max_pixels=max_pixels)

    if ratio < 1.0:
        # Scale each image by the same ratio, preserving each image's own aspect/size baseline.
        images = [
            img.resize(
                (max(1, int(img.width * ratio)), max(1, int(img.height * ratio))),
                resample=Image.Resampling.LANCZOS,
            )
            for img in images
        ]

    cropped_images = []
    for img in images:
        cropped_img, cropped_h, cropped_w = center_crop_to_multiple(img, multiple=32)
        cropped_images.append(cropped_img)

    cond_images, _ = pipe.prepare_images(cropped_images, "condition", reshape=False)
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            image=cond_images,
            prompt=[data["prompt"][0]],
            device=device,
        )

    vae_images, _ = pipe.prepare_images(cropped_images, "vae", reshape=False)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        image_latents_list = [
            pipe._encode_vae_image(img.to(device=device, dtype=torch.bfloat16), generator=None)
            for img in vae_images
        ]
        image_latents = torch.cat(image_latents_list, dim=2)

    return prompt_embeds, prompt_attention_mask, image_latents, cropped_h, cropped_w, ratio


def build_dataset_from_args(args):
    if args.dataset_config:
        return MultiSourceT5Dataset(
            config_path=args.dataset_config,
            split_name=args.split_name,
            default_base_image_path=args.base_image_path,
            global_seed=args.seed,
            strict_unique_filename=not args.allow_duplicate_filenames,
        )
    if not args.prompt_dir:
        raise ValueError("either --dataset_config or --prompt_dir must be provided")
    return T5dataset(
        jsonl_path=args.prompt_dir,
        base_image_path=args.base_image_path,
        source_name="single_source",
    )


def main(args):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "rank", rank, "local_rank", local_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_attention_mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "image_latents"), exist_ok=True)

    train_dataset = build_dataset_from_args(args)
    if rank == 0:
        print(f"[preprocess] total selected samples: {len(train_dataset)}")

    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    pipeline = QwenImageEditPlusPipeline.from_pretrained(args.pretrained_model_name_or_path)
    modules = QwenImageEditModules.from_pipeline(pipeline)
    if args.vae_model_path is not None:
        modules.vae = load_vae_from_local_dir(args.vae_model_path)
        print(f"[preprocess] loaded custom VAE from {args.vae_model_path}")
    pipeline = QwenImageEditPlusPipeline(**modules.to_dict()).to(device=device, dtype=torch.bfloat16)
    print("QwenImageEditPlus pipeline with custom VAE loaded")

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=rank != 0):
        with torch.inference_mode(), torch.autocast("cuda"):
            prompt_embeds, prompt_attention_mask, image_latents, cropped_h, cropped_w, resize_ratio = calculate_embeds(
                pipeline, data, device, args.max_pixels
            )

            original_length = prompt_embeds.shape[1]
            target_length = 1500
            pad_len = target_length - original_length
            prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, pad_len), "constant", 0)
            prompt_attention_mask = F.pad(prompt_attention_mask, (0, pad_len), "constant", 0)

            for idx, video_name in enumerate(data["filename"]):
                prompt_embed_path = os.path.join(args.output_dir, "prompt_embed", video_name + ".pt")
                prompt_attention_mask_path = os.path.join(args.output_dir, "prompt_attention_mask", video_name + ".pt")
                image_latents_path = os.path.join(args.output_dir, "image_latents", video_name + ".pt")

                torch.save(prompt_embeds[idx], prompt_embed_path)
                torch.save(prompt_attention_mask[idx], prompt_attention_mask_path)
                torch.save(image_latents[idx], image_latents_path)

                item = {
                    "prompt_embed_path": video_name + ".pt",
                    "prompt_attention_mask": video_name + ".pt",
                    "image_latents": video_name + ".pt",
                    "prompt": data["prompt"][idx],
                    "original_length": original_length,
                    "resize_ratio": float(resize_ratio),
                    "calculated_height": int(cropped_h),
                    "calculated_width": int(cropped_w),
                    "source_dataset": data.get("source_dataset", ["unknown"])[idx],
                }
                json_data.append(item)

    dist.barrier()
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, json_data)
    if rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "preprocess.json"), "w", encoding="utf-8") as f:
            json.dump(all_json_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--vae_model_path", type=str, default=None)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size per device. This script is designed for batch_size=1.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vae_debug", action="store_true")

    # legacy single-source mode
    parser.add_argument("--prompt_dir", type=str, default="")
    parser.add_argument(
        "--base_image_path",
        type=str,
        default="",
        help="Root directory for relative image paths in JSONL.",
    )

    # mixed-source mode
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="",
        help="Path to mixed dataset config (.json/.jsonc).",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Split name in dataset config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for sampling in mixed dataset mode.",
    )
    parser.add_argument(
        "--allow_duplicate_filenames",
        action="store_true",
        help="Allow duplicate sample keys (will overwrite files if duplicated).",
    )

    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help="Max total pixels for preprocessing images. Images are uniformly downscaled if exceeded.",
    )

    args = parser.parse_args()
    main(args)