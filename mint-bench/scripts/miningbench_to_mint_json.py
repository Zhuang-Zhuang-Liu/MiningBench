#!/usr/bin/env python3
"""Convert MiningBench-v1 benchmark.jsonl into MINT JSON Lines task format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reference_from_ground_truth(ground_truth: dict) -> str:
    features = ", ".join(ground_truth.get("answer_features", []))
    rule = ground_truth.get("answer_rule", "")
    return (
        f"<answer_features>{features}</answer_features>\n"
        f"<answer_rule>{rule}</answer_rule>"
    )


def convert(input_path: Path, output_path: Path, data_root: Path) -> int:
    rows_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            source = json.loads(line)
            rel_data_path = source["input_data_path"]
            abs_data_path = data_root / Path(rel_data_path).relative_to(data_root.name)
            prompt = source["question"].replace(
                f'文件路径："{rel_data_path}"',
                f'文件路径："{abs_data_path}"',
            )
            prompt = prompt.replace(f"`{rel_data_path}`", f"`{abs_data_path}`")

            item = {
                "id": source["id"],
                "prompt": prompt,
                "reference": reference_from_ground_truth(source["ground_truth"]),
                "source_input_data_path": rel_data_path,
                "input_data_path": str(abs_data_path),
                "difficulty": source.get("difficulty"),
                "split": source.get("split"),
                "ground_truth": source.get("ground_truth"),
                "metadata": source.get("metadata"),
            }
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            rows_written += 1

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MiningBench-v1 tasks into MINT JSON Lines format."
    )
    parser.add_argument("--input", required=True, help="MiningBench benchmark.jsonl")
    parser.add_argument(
        "--output",
        required=True,
        help="Output MINT JSON Lines path; MINT convention is test_prompts.json",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Absolute MiningBench-v1 directory containing task_*/data.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()

    rows_written = convert(input_path, output_path, data_root)
    print(f"Written rows: {rows_written}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
