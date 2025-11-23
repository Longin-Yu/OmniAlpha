
import os, json5, json
import shutil
from typing import *
from pathlib import Path
import fnmatch
import torch


BackgroundType = Union[float, torch.Tensor]

def alpha_blend(fg: torch.Tensor, bg: BackgroundType, dim: int = -3, alpha_min: int = 0, alpha_max: int = 1) -> torch.Tensor:
    assert fg.shape[dim] == 4, "The specified dimension must have size 4 (RGBA channels)."
    assert isinstance(bg, float) or bg.shape[dim] == 3, "Background must be a float or have 3 channels (RGB)."
    rgb, alpha = torch.split(fg, [3, 1], dim=dim)
    # print(f'{rgb.shape=}, {alpha.shape=}')
    denormalized_alpha = (alpha - alpha_min) / (alpha_max - alpha_min)
    return rgb * denormalized_alpha + (1 - denormalized_alpha) * bg

def load_json_file(path: str) -> Any:
    assert path.endswith('.jsonl') \
        or path.endswith('.json') \
        or path.endswith('.jsonc'), \
        "data_path must be a jsonl/json/jsonc file."
    
    with open(path, 'r') as f:
        if path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        elif path.endswith('.jsonc'):
            return json5.load(f)
        else:
            return json.load(f)

def load_gitignore_patterns(gitignore_path: Path) -> List[str]:
    patterns = []
    if gitignore_path.exists():
        with gitignore_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
    return patterns

def is_ignored(path: str, patterns: List[str], root_dir: str) -> bool:
    rel_path = os.path.relpath(path, root_dir)
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(os.path.basename(rel_path), pattern):
            return True
    return False

def copy_code_files(source_dir: str, target_dir: str, excludes: Optional[List[str]] = None):
    source_dir = os.path.abspath(source_dir)
    target_dir = os.path.abspath(target_dir)

    # 处理排除规则
    if excludes is not None:
        ignore_patterns = excludes
    else:
        gitignore_path = Path(source_dir) / ".gitignore"
        if gitignore_path.exists():
            ignore_patterns = load_gitignore_patterns(gitignore_path)
        else:
            ignore_patterns = ['output', 'outputs', 'save', 'saves', 'wandb', 'log', 'logs']

    for root, dirs, files in os.walk(source_dir):
        # 跳过符号链接目录
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]

        # 如果当前路径是 target_dir 或其子目录，跳过
        if os.path.commonpath([root, target_dir]) == target_dir:
            print(f"Skipping target_dir or its subdir: {root}")
            continue

        # 跳过被忽略的目录
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d), ignore_patterns, source_dir)]

        for file in files:
            if not (file.endswith('.py') or file.endswith('.sh')):
                continue

            source_path = os.path.join(root, file)
            if is_ignored(source_path, ignore_patterns, source_dir):
                continue

            relative_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, relative_path)

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)
            # print in cyan
            # print(f"\033[96mCopied: {source_path} -> {target_path}\033[0m")

# 示例用法：
# copy_code_files('/path/to/source', '/path/to/target')
# 或带排除规则：
# copy_code_files('/path/to/source', '/path/to/target', excludes=['*.ipynb', 'outputs/', 'wandb/'])
