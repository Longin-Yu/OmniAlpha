# OmniAlpha: A Sequence-to-Sequence Framework for Unified Multi-Task RGBA Generation

---

**This is the official repository for "OmniAlpha: A Sequence-to-Sequence Framework for Unified Multi-Task RGBA Generation".**

![examples](assets/examples_01.png)

---

## 📂 File Organization

The project structure is organized as follows:

```
.
├── alpha        # package root
├── configs      # configuration files
├── scripts      # bash scripts
├── setup.py     # package definitions
└── tasks        # python/jupyter scripts
````

## 📦 Installation

```bash
conda create -n OmniAlpha python=3.10
conda activate OmniAlpha
pip install -r requirements.txt --user
pip install -e .
```

## 📄 Data Preparation

> 📂 Please refer to `configs/datasets.jsonc` for dataset configuration.
> Each dataset entry consists of two required fields:
>
>   * `data_path`: Path to the JSON annotation file.
>   * `image_dir`: Root directory for the dataset images.

The dataset file, specified by the `data_path` key in `configs/datasets.jsonc`, should be a JSONL file adhering to the following structure. Please ensure that both input_images and output_images are provided as relative paths:

```jsonl
{"id": "case_0", "prompt": "Vintage camera next to a brown glass bottle.", "input_images": ["images_512/case_0/base.png"], "output_images": ["images_512/case_0/00.png"]}
{"id": "case_1", "prompt": "A vintage-style globe with a map of North and South America, mounted on a black stand.;Antique key with ornate design, attached to a chain.", "input_images": ["images_512/case_1/base.png"], "output_images": ["images_512/case_1/00.png", "images_512/case_1/01.png"]}
...
```

## 🔽 Model Download

Pretrained model checkpoints are **Coming Soon**. Stay tuned!

## 🚀 Inference

To run inference using pretrained models:

```bash
# Run inference
```

## 🏋️ Training

Run the following command to start training the Qwen-based image model:

```bash
# Start training
```

## Citation

```bibtex
coming soon
```
