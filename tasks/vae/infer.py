import os
import argparse
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from accelerate import Accelerator
from alpha.vae.modeling import load_vae_from_local_dir, AutoencoderClass
from finetune import LocalImageDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_vae_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["float32", "fp16", "bf16"])
    return parser.parse_args()


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    file_paths = [example["file_path"] for example in examples]
    return {"pixel_values": pixel_values, "file_path": file_paths}


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir) or len(os.listdir(args.input_dir)) == 0:
        print(f"No images found in ${args.input_dir}.")
        return
    
    dtype_map = {
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    weight_dtype = dtype_map[args.dtype]

    accelerator = Accelerator()
    device = accelerator.device

    os.makedirs(args.output_dir, exist_ok=True)

    # Define image preprocessing transforms
    test_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Lambda(lambda x: x.unsqueeze(1))
        ]
    )

    dataset = LocalImageDataset(args.input_dir, test_transforms)
    print(f"dataset length: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    vae = load_vae_from_local_dir(args.pretrained_vae_path).to(device, dtype=weight_dtype)
    vae.eval()

    vae, dataloader = accelerator.prepare(vae, dataloader)

    for batch in dataloader:
        with torch.no_grad():
            with accelerator.autocast():
                x = batch["pixel_values"].to(dtype=weight_dtype)
                recon = vae(x).sample
                recon = (recon * 0.5 + 0.5).clamp(0, 1)

        # Save each reconstructed image
        for i, file_path in enumerate(batch["file_path"]):
            save_path = os.path.join(args.output_dir, f"{os.path.basename(file_path)}")
            save_image(recon[i].cpu(), save_path)

    accelerator.print("✅ Inference complete.")


if __name__ == "__main__":
    main()
