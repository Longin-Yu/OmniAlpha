import os
import json
import time
import math
import uuid
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import gradio as gr
import torch
from PIL import Image, ImageOps

from alpha.vae.modeling import load_vae_from_local_dir
from alpha.pipelines.qwen_image_edit import CustomQwenImageEditPlusPipeline as QwenImageEditPlusPipeline


TOTAL_CANVAS_PIXELS = 1024 * 1024
PIPELINE = None
PIPELINE_CACHE_KEY = None


@dataclass
class InputSlotConfig:
    key: str
    label: str
    tip: str


@dataclass
class TaskConfig:
    value: str
    label: str
    description: str
    fixed_prompt: str
    frames: int
    needs_text_input: bool
    text_label: str
    text_placeholder: str
    input_slots: List[InputSlotConfig]
    fixed_output_size: Optional[Tuple[int, int]] = None
    max_pixels: Optional[int] = None


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "t2i": TaskConfig(
        value="t2i",
        label="t2i",
        description="文本生成透明背景 RGBA 图像，不需要上传输入图像，固定输出 1024 × 1024。",
        fixed_prompt="Generate a transparent-background RGBA image.",
        frames=1,
        needs_text_input=True,
        text_label="图像描述",
        text_placeholder="例如：A man with short brown hair and a dark gray t-shirt is seen from behind, his head slightly turned to the side.",
        input_slots=[],
        fixed_output_size=(1024, 1024),
    ),
    "ObjectClear": TaskConfig(
        value="ObjectClear",
        label="ObjectClear",
        description="使用第一张图作为源场景，第二张图作为目标掩码，移除被掩码标记的目标与其影响。",
        fixed_prompt=(
            "Use the first image as the source scene and the second image as the object mask. "
            "Remove the masked object and all of its associated effects, including shadows, reflections, "
            "highlights, contact traces, and residual artifacts, even when these effects extend beyond the mask. "
            "Reconstruct the clean base background as if the object had never been present."
        ),
        frames=1,
        needs_text_input=False,
        text_label="",
        text_placeholder="",
        input_slots=[
            InputSlotConfig("source", "源场景图", "第 1 张图，待清除目标的原始场景。"),
            InputSlotConfig("mask", "目标掩码图", "第 2 张图，标出需要移除的对象区域。"),
        ],
        max_pixels=TOTAL_CANVAS_PIXELS // 3,
    ),
    "automatting": TaskConfig(
        value="automatting",
        label="automatting",
        description="自动抠出前景并保留精细透明度，输入 1 张，输出 1 张。",
        fixed_prompt=(
            "Automatically matte this image and extract the foreground with a physically accurate alpha channel "
            "that preserves true transparency and fine details."
        ),
        frames=1,
        needs_text_input=False,
        text_label="",
        text_placeholder="",
        input_slots=[InputSlotConfig("input", "待抠图图像", "上传需要自动抠图的原始图像。")],
        max_pixels=TOTAL_CANVAS_PIXELS // 2,
    ),
    "refmatting": TaskConfig(
        value="refmatting",
        label="refmatting",
        description="根据文本描述，从图像中提取目标并保留细节与透明区域，输入 1 张，输出 1 张。",
        fixed_prompt="Extract the object described by the text, preserving fine details and transparency:",
        frames=1,
        needs_text_input=True,
        text_label="目标文本描述",
        text_placeholder="例如：the salient female mankind with the black knit farthest to the left of the picture",
        input_slots=[InputSlotConfig("input", "参考图像", "上传需要按文本提取目标的原始图像。")],
        max_pixels=TOTAL_CANVAS_PIXELS // 2,
    ),
    "layerdecompose": TaskConfig(
        value="layerdecompose",
        label="layerdecompose",
        description="将单张图像拆分为从后到前的完整图层序列，输入 1 张，输出 2 张。",
        fixed_prompt=(
            "Decompose this image into an ordered back-to-front layer sequence: output the complete background first, "
            "then output complete RGBA foreground layers in depth order, where each foreground layer preserves the "
            "entire object rather than only its visible region, and the original image must be exactly reconstructed "
            "by sequentially alpha-compositing these layers one by one from back to front."
        ),
        frames=2,
        needs_text_input=False,
        text_label="",
        text_placeholder="",
        input_slots=[InputSlotConfig("input", "待分层图像", "上传需要做层分解的输入图像。")],
        max_pixels=TOTAL_CANVAS_PIXELS // 3,
    ),
}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(SCRIPT_DIR, "omnialpha")


def maybe_example(path: str):
    return path if os.path.exists(path) else None


EXAMPLES = [
    [
        "t2i",
        "A man with short brown hair and a dark gray t-shirt is seen from behind, his head slightly turned to the side.",
        None,
        None,
        42,
        50,
        4.0,
        1,
    ],
    [
        "ObjectClear",
        "",
        maybe_example(os.path.join(EXAMPLE_DIR, "objclr_ori.png")),
        maybe_example(os.path.join(EXAMPLE_DIR, "objclr_mask.png")),
        42,
        50,
        4.0,
        1,
    ],
    [
        "automatting",
        "",
        maybe_example(os.path.join(EXAMPLE_DIR, "automatting.png")),
        None,
        42,
        50,
        4.0,
        1,
    ],
    [
        "refmatting",
        "a man with sunglasses and white shirt located on the left side of the photo",
        maybe_example(os.path.join(EXAMPLE_DIR, "refmatte.png")),
        None,
        42,
        50,
        4.0,
        1,
    ],
    [
        "layerdecompose",
        "",
        maybe_example(os.path.join(EXAMPLE_DIR, "decompose.png")),
        None,
        42,
        50,
        4.0,
        1,
    ],
]


def parse_args():
    parser = argparse.ArgumentParser(description="OmniAlpha Gradio App")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_vae_model", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_dir", type=str, default="./outputs_gradio")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def build_pipeline(model_path: str, vae_path: Optional[str], lora_path: Optional[str], device: str, dtype: torch.dtype):
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


def get_pipeline(args):
    global PIPELINE, PIPELINE_CACHE_KEY
    cache_key = (
        args.pretrained_model_name_or_path,
        args.pretrained_vae_model,
        args.lora_path,
        args.device,
        args.dtype,
    )
    if PIPELINE is None or PIPELINE_CACHE_KEY != cache_key:
        torch_dtype = get_torch_dtype(args.dtype)
        PIPELINE = build_pipeline(
            model_path=args.pretrained_model_name_or_path,
            vae_path=args.pretrained_vae_model,
            lora_path=args.lora_path,
            device=args.device,
            dtype=torch_dtype,
        )
        PIPELINE_CACHE_KEY = cache_key
    return PIPELINE


def format_dimensions(width: int, height: int) -> str:
    return f"{width} × {height}"


def format_pixels(pixels: int) -> str:
    return f"{pixels:,} px"


def build_task_prompt(task: str, text_input: str) -> str:
    cfg = TASK_CONFIGS[task]
    text_input = (text_input or "").strip()
    if task in {"t2i", "refmatting"} and text_input:
        return f"{cfg.fixed_prompt} {text_input}"
    return cfg.fixed_prompt


def compute_constrained_size(original_width: int, original_height: int, max_pixels: int) -> Tuple[int, int]:
    scale = min(1.0, math.sqrt(max_pixels / float(original_width * original_height)))
    width = max(32, int(original_width * scale) // 32 * 32)
    height = max(32, int(original_height * scale) // 32 * 32)
    while width * height > max_pixels and (width > 32 or height > 32):
        if width >= height and width > 32:
            width -= 32
        elif height > 32:
            height -= 32
        else:
            break
    return width, height


def preprocess_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    image = image.convert("RGBA")
    fitted = ImageOps.fit(
        image,
        (target_width, target_height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    return fitted


def image_meta_dict(img: Image.Image, processed: Image.Image, max_pixels: int) -> Dict[str, Any]:
    ow, oh = img.size
    pw, ph = processed.size
    original_pixels = ow * oh
    processed_pixels = pw * ph
    exceeded = original_pixels > max_pixels
    transformed = (ow != pw) or (oh != ph)
    status = "尺寸已满足当前任务约束，无需额外压缩。"
    if exceeded:
        status = f"原始像素超过任务上限，已前端压缩到 {format_pixels(processed_pixels)}。"
    elif transformed:
        status = "已按 32 倍数要求进行居中裁剪。"
    return {
        "original_width": ow,
        "original_height": oh,
        "original_pixels": original_pixels,
        "processed_width": pw,
        "processed_height": ph,
        "processed_pixels": processed_pixels,
        "transformed": transformed,
        "status": status,
    }


def validate_and_prepare_inputs(task: str, image1: Optional[Image.Image], image2: Optional[Image.Image]):
    cfg = TASK_CONFIGS[task]
    processed_images: List[Image.Image] = []
    meta_lines: List[str] = []

    if cfg.fixed_output_size:
        width, height = cfg.fixed_output_size
        resolution_summary = f"固定输出 {format_dimensions(width, height)}"
        resolution_hint = f"t2i 固定输出 {format_pixels(width * height)}，不允许修改分辨率。"
        return processed_images, width, height, resolution_summary, resolution_hint, ""

    if task == "ObjectClear":
        if image1 is None or image2 is None:
            raise gr.Error("ObjectClear 需要上传 2 张图：源场景图和目标掩码图。")
        if image1.size != image2.size:
            raise gr.Error(
                f"ObjectClear 要求两张输入图原始尺寸完全一致。当前源图为 {format_dimensions(*image1.size)}，掩码图为 {format_dimensions(*image2.size)}。"
            )
        width, height = compute_constrained_size(image1.size[0], image1.size[1], cfg.max_pixels)
        p1 = preprocess_image(image1, width, height)
        p2 = preprocess_image(image2, width, height)
        processed_images = [p1, p2]
        m1 = image_meta_dict(image1, p1, cfg.max_pixels)
        m2 = image_meta_dict(image2, p2, cfg.max_pixels)
        meta_lines.extend([
            f"源场景图：原始 {format_dimensions(m1['original_width'], m1['original_height'])} / 提交 {format_dimensions(m1['processed_width'], m1['processed_height'])} / {m1['status']}",
            f"目标掩码图：原始 {format_dimensions(m2['original_width'], m2['original_height'])} / 提交 {format_dimensions(m2['processed_width'], m2['processed_height'])} / {m2['status']}",
        ])
    else:
        if image1 is None:
            raise gr.Error(f"{cfg.label} 需要上传 1 张输入图。")
        width, height = compute_constrained_size(image1.size[0], image1.size[1], cfg.max_pixels)
        p1 = preprocess_image(image1, width, height)
        processed_images = [p1]
        m1 = image_meta_dict(image1, p1, cfg.max_pixels)
        meta_lines.append(
            f"输入图：原始 {format_dimensions(m1['original_width'], m1['original_height'])} / 提交 {format_dimensions(m1['processed_width'], m1['processed_height'])} / {m1['status']}"
        )

    resolution_summary = f"当前提交尺寸 {format_dimensions(width, height)}"
    resolution_hint = f"像素预算上限 {format_pixels(cfg.max_pixels)}。前端已自动等比例缩放、居中裁剪，并保证长宽都是 32 的倍数。"
    return processed_images, width, height, resolution_summary, resolution_hint, "\n".join(meta_lines)


def save_outputs(output_dir: str, task_id: str, images: List[Image.Image], final_format: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for idx, image in enumerate(images):
        if final_format == "RGBA":
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
        path = os.path.join(output_dir, f"{task_id}_{idx}.png")
        image.save(path)
        saved.append(path)
    return saved




def run_generation(
    task: str,
    text_input: str,
    image1: Optional[Image.Image],
    image2: Optional[Image.Image],
    seed: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    num_images_per_prompt: int,
    runtime_args,
):
    cfg = TASK_CONFIGS[task]
    prompt = build_task_prompt(task, text_input)
    processed_images, width, height, resolution_summary, resolution_hint, input_meta = validate_and_prepare_inputs(
        task, image1, image2
    )

    torch_dtype = get_torch_dtype(runtime_args.dtype)
    pipeline = build_pipeline(
        model_path=runtime_args.pretrained_model_name_or_path,
        vae_path=runtime_args.pretrained_vae_model,
        lora_path=runtime_args.lora_path,
        device=runtime_args.device,
        dtype=torch_dtype,
    )
    final_format = "RGBA" if runtime_args.pretrained_vae_model else "RGB"
    task_id = str(uuid.uuid4())[:8]
    generator = torch.Generator(runtime_args.device).manual_seed(int(seed))

    infer_kwargs = {
        "prompt": prompt,
        "negative_prompt": "",
        "num_inference_steps": int(num_inference_steps),
        "height": int(height),
        "width": int(width),
        "true_cfg_scale": float(true_cfg_scale),
        "generator": generator,
        "num_images_per_prompt": int(num_images_per_prompt),
        "frames": int(cfg.frames),
    }
    if processed_images:
        infer_kwargs["image"] = processed_images

    start = time.time()
    with torch.no_grad():
        outputs = pipeline(**infer_kwargs).images
    if not outputs:
        raise gr.Error("生成失败：outputs 为空")

    saved_paths = save_outputs(runtime_args.output_dir, task_id, outputs, final_format)
    elapsed = time.time() - start

    result = {
        "task_id": task_id,
        "images": saved_paths,
        "num_input_images": len(processed_images),
        "gen_time": elapsed,
        "machine": "NVIDIA A100 80GB PCIe",
        "seed": int(seed),
        "prompt": prompt,
        "negative_prompt": "",
        "num_images": len(saved_paths),
        "num_images_per_prompt": int(num_images_per_prompt),
        "frames": int(cfg.frames),
        "image_format": final_format,
        "output_resolution": f"{width} × {height}",
        "task": task,
    }

    json_path = os.path.join(runtime_args.output_dir, f"{task_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    info_md = f"""
### 编辑结果
- **任务类型**：{cfg.label}
- **任务 ID**：{task_id}
- **生成用时**：{elapsed:.2f} 秒
- **使用设备**：NVIDIA A100 80GB PCIe
- **输出分辨率**：{width} × {height}
- **输入图像数**：{len(processed_images)}
- **结果张数**：{len(saved_paths)}
- **输出格式**：{final_format}
- **提示词**：{prompt}

### 自动分辨率
- **摘要**：{resolution_summary}
- **说明**：{resolution_hint}

### 输入图像信息
{input_meta if input_meta else '当前任务不需要输入图像。'}

### 输出文件
- 图片目录：`{runtime_args.output_dir}`
- 结果 JSON：`{json_path}`
"""

    gallery_items = saved_paths
    return gallery_items, info_md, resolution_summary, resolution_hint


def update_task_ui(task: str):
    cfg = TASK_CONFIGS[task]
    prompt_preview = build_task_prompt(task, "")

    if cfg.fixed_output_size:
        width, height = cfg.fixed_output_size
        resolution_summary = f"固定输出 {format_dimensions(width, height)}"
        resolution_hint = f"t2i 固定输出 {format_pixels(width * height)}，不允许修改分辨率。"
    else:
        resolution_summary = "上传后自动计算提交尺寸"
        resolution_hint = f"像素预算上限 {format_pixels(cfg.max_pixels)}。上传图像后会自动压缩并裁剪到 32 的倍数。"

    slot1_visible = len(cfg.input_slots) >= 1
    slot2_visible = len(cfg.input_slots) >= 2
    text_visible = cfg.needs_text_input

    slot1_label = cfg.input_slots[0].label if slot1_visible else "输入图像 1"
    slot2_label = cfg.input_slots[1].label if slot2_visible else "输入图像 2"
    slot1_tip = cfg.input_slots[0].tip if slot1_visible else ""
    slot2_tip = cfg.input_slots[1].tip if slot2_visible else ""

    return (
        gr.update(value=f"### {cfg.label}\n{cfg.description}\n\n**固定指令**：{prompt_preview}"),
        gr.update(
            visible=text_visible,
            label=cfg.text_label or "文本输入",
            placeholder=cfg.text_placeholder or "请输入文本",
        ),
        gr.update(visible=slot1_visible, label=slot1_label),
        gr.update(value=slot1_tip, visible=slot1_visible),
        gr.update(visible=slot2_visible, label=slot2_label),
        gr.update(value=slot2_tip, visible=slot2_visible),
        gr.update(value=f"**{resolution_summary}**\n\n{resolution_hint}"),
    )


def build_demo(runtime_args):
    custom_css = """
    .gradio-container {background: #e6f7ff !important;}
    .oa-result-wrap {
        min-height: 560px;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        background: #ffffff;
        padding: 14px;
        overflow: auto;
    }
    .oa-result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 14px;
        align-items: start;
    }
    .oa-result-card {
        border: 1px solid #ececec;
        border-radius: 12px;
        overflow: hidden;
        background: #fafafa;
        display: flex;
        flex-direction: column;
    }
    .oa-result-link {
        display: block;
        width: 100%;
        height: 260px;
        background: #fff;
    }
    .oa-result-image {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
        background: #fff;
    }
    .oa-result-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 12px;
        font-size: 13px;
    }
    .oa-download-link {
        color: #2563eb;
        text-decoration: none;
        font-weight: 600;
    }
    .oa-empty {
        min-height: 520px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #999;
        font-size: 16px;
    }
    @media (max-width: 900px) {
        .oa-result-grid {
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        }
        .oa-result-link {
            height: 180px;
        }
    }
    """

    with gr.Blocks(title="OmniAlpha 多图层图像编辑") as demo:
        gr.Markdown("# OmniAlpha 多图层图像编辑")

        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                gr.Markdown("### 参数控制")
                task = gr.Dropdown(
                    choices=[(cfg.label, cfg.value) for cfg in TASK_CONFIGS.values()],
                    value="t2i",
                    label="任务类型",
                )
                task_desc = gr.Markdown()
                text_input = gr.Textbox(label="图像描述", lines=4, placeholder=TASK_CONFIGS["t2i"].text_placeholder)
                image1_tip = gr.Markdown(visible=False)
                image1 = gr.Image(type="pil", label="输入图像 1", visible=False, height=220)
                image2_tip = gr.Markdown(visible=False)
                image2 = gr.Image(type="pil", label="输入图像 2", visible=False, height=220)
                resolution_md = gr.Markdown()

                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="随机种子")
                    num_inference_steps = gr.Number(value=50, precision=0, label="推理步数")

                with gr.Row():
                    true_cfg_scale = gr.Number(value=4.0, label="CFG Scale")
                    num_images_per_prompt = gr.Number(value=1, precision=0, label="输出张数")

                run_btn = gr.Button("开始执行", variant="primary")

            with gr.Column(scale=3):
                gr.Markdown("### 示例模板")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[task, text_input, image1, image2, seed, num_inference_steps, true_cfg_scale, num_images_per_prompt],
                    label="点击示例可快速填充任务类型与常用参数",
                )
                gr.Markdown("### 自动分辨率")
                resolution_summary = gr.Textbox(label="摘要", interactive=False)
                resolution_hint = gr.Textbox(label="说明", interactive=False, lines=3)

        with gr.Row():
            with gr.Column():
                gallery = gr.Gallery(
                    label="编辑结果",
                    columns=3,
                    height=560,
                    object_fit="contain",
                    preview=True,
                )
                result_info = gr.Markdown("暂无编辑结果")

        task.change(
            fn=update_task_ui,
            inputs=[task],
            outputs=[task_desc, text_input, image1, image1_tip, image2, image2_tip, resolution_md],
        )

        run_btn.click(
            fn=lambda task, text_input, image1, image2, seed, num_inference_steps, true_cfg_scale, num_images_per_prompt: run_generation(
                task,
                text_input,
                image1,
                image2,
                int(seed),
                int(num_inference_steps),
                float(true_cfg_scale),
                int(num_images_per_prompt),
                runtime_args,
            ),
            inputs=[task, text_input, image1, image2, seed, num_inference_steps, true_cfg_scale, num_images_per_prompt],
            outputs=[gallery, result_info, resolution_summary, resolution_hint],
        )

        demo.load(
            fn=update_task_ui,
            inputs=[task],
            outputs=[task_desc, text_input, image1, image1_tip, image2, image2_tip, resolution_md],
        )

    return demo, custom_css


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    demo, custom_css = build_demo(args)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        css=custom_css,
        theme=gr.themes.Soft(),
    )
