import os
import json
import time
import argparse
from typing import List, Optional

import torch
from PIL import Image

from alpha.vae.modeling import load_vae_from_local_dir
from alpha.pipelines.qwen_image_edit import CustomQwenImageEditPlusPipeline as QwenImageEditPlusPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="QwenImage Multimodal Inference")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_model", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)

    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument(
        "--input_images",
        type=str,
        nargs="*",
        default=[],
        help="Optional input image paths. Can pass 0 to n images.",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_id", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--frames", type=int, default=1)

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def build_pipeline(
    model_path: str,
    vae_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    if vae_path:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            vae=load_vae_from_local_dir(vae_path),
        )
    else:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path)

    if lora_path:
        pipeline.load_lora_weights(lora_path)

    pipeline = pipeline.to(device, dtype)
    return pipeline


def load_input_images(image_paths: List[str]) -> List[Image.Image]:
    images = []
    for path in image_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input image not found: {path}")
        img = Image.open(path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        images.append(img)
    return images


def main():
    start_time = time.time()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch_dtype = get_torch_dtype(args.dtype)

    print("🎯 Parameters:")
    print(f"  --pretrained_model_name_or_path : {args.pretrained_model_name_or_path}")
    print(f"  --pretrained_vae_model         : {args.pretrained_vae_model}")
    print(f"  --lora_path                    : {args.lora_path}")
    print(f"  --prompts                      : {args.prompts}")
    print(f"  --negative_prompt              : {args.negative_prompt}")
    print(f"  --input_images                 : {args.input_images}")
    print(f"  --output_dir                   : {args.output_dir}")
    print(f"  --task_id                      : {args.task_id}")
    print(f"  --seed                         : {args.seed}")
    print(f"  --num_images_per_prompt        : {args.num_images_per_prompt}")
    print(f"  --frames                       : {args.frames}")
    print(f"  --height                       : {args.height}")
    print(f"  --width                        : {args.width}")
    print(f"  --num_inference_steps          : {args.num_inference_steps}")
    print(f"  --true_cfg_scale               : {args.true_cfg_scale}")
    print(f"  --device                       : {args.device}")
    print(f"  --dtype                        : {args.dtype}")

    final_format = "RGBA" if args.pretrained_vae_model else "RGB"

    print("🚀 Loading pipeline...")
    pipeline = build_pipeline(
        model_path=args.pretrained_model_name_or_path,
        vae_path=args.pretrained_vae_model,
        lora_path=args.lora_path,
        device=args.device,
        dtype=torch_dtype,
    )

    input_images = load_input_images(args.input_images) if args.input_images else []

    generator = torch.Generator(args.device).manual_seed(args.seed)

    print("🚀 Generating images...")
    infer_kwargs = {
        "prompt": args.prompts,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "true_cfg_scale": args.true_cfg_scale,
        "generator": generator,
        "num_images_per_prompt": args.num_images_per_prompt,
        "frames": args.frames,
    }

    if len(input_images) > 0:
        infer_kwargs["image"] = input_images

    with torch.no_grad():
        outputs = pipeline(**infer_kwargs).images

    if not outputs:
        raise RuntimeError("Generation failed: outputs is empty")

    saved_images = []
    for idx, image in enumerate(outputs):
        if final_format == "RGBA" and image.mode != "RGBA":
            image = image.convert("RGBA")

        save_path = os.path.join(args.output_dir, f"{args.task_id}_{idx}.png")
        image.save(save_path)
        saved_images.append(save_path)

    elapsed_time = time.time() - start_time

    result = {
        "task_id": args.task_id,
        "images": saved_images,
        "input_images": args.input_images,
        "num_input_images": len(args.input_images),
        "gen_time": elapsed_time,
        "machine": "NVIDIA A100 80GB PCIe",
        "seed": args.seed,
        "prompt": args.prompts,
        "negative_prompt": args.negative_prompt,
        "num_images": len(saved_images),
        "num_images_per_prompt": args.num_images_per_prompt,
        "frames": args.frames,
        "image_format": final_format,
    }

    json_path = os.path.join(args.output_dir, f"{args.task_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("✅ Generation finished")
    print(f"📁 Images saved to: {args.output_dir}")
    print(f"📁 Json saved to: {json_path}")
    print(f"⏱️ Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()