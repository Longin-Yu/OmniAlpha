# python tasks/rl/dedup_refmatte_jsonl.py --jsonl_path /path/to/RefMatte_15k_base_refmatting.jsonl --output_path /path/to/RefMatte_15k_base_refmatting.dedup.jsonl

import argparse
import json
import os
import shutil
from typing import List, Tuple


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_idx}: {e}") from e
    return rows


def key_by_input0(rec: dict) -> str:
    input_images = rec.get("input_images")
    if not isinstance(input_images, list) or len(input_images) == 0:
        return ""
    return str(input_images[0])


def dedup_keep_first(rows: List[dict]) -> Tuple[List[dict], int]:
    seen = set()
    deduped = []
    dup_count = 0
    for rec in rows:
        k = key_by_input0(rec)
        if k in seen:
            dup_count += 1
            continue
        seen.add(k)
        deduped.append(rec)
    return deduped, dup_count


def dump_jsonl(path: str, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Path to RefMatte jsonl.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Output path for deduplicated jsonl. If empty, create '<input>.dedup.jsonl'.",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Disable backup when explicitly running in-place.",
    )
    args = parser.parse_args()

    src = args.jsonl_path
    if args.output_path:
        dst = args.output_path
    else:
        root, ext = os.path.splitext(args.jsonl_path)
        if not ext:
            ext = ".jsonl"
        dst = root + ".dedup" + ext

    rows = load_jsonl(src)
    deduped, dup_count = dedup_keep_first(rows)

    if dst == src and (not args.no_backup):
        backup = src + ".bak"
        shutil.copy2(src, backup)
        print(f"[dedup] backup created: {backup}")

    if dst != src:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    dump_jsonl(dst, deduped)

    print(f"[dedup] input rows   : {len(rows)}")
    print(f"[dedup] output rows  : {len(deduped)}")
    print(f"[dedup] removed rows : {dup_count}")
    print(f"[dedup] saved to     : {dst}")


if __name__ == "__main__":
    main()
