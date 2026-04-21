#!/usr/bin/env python3
"""Convert MINT-style results.jsonl into a readable CSV and summary metrics."""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


SOLUTION_PATTERN = re.compile(r"<solution>\s*(.*?)\s*</solution>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a MINT-style results.jsonl file to CSV and compute summary metrics."
    )
    parser.add_argument("input_jsonl", help="Path to input results.jsonl")
    parser.add_argument(
        "--output-csv",
        help="Path to output CSV. Defaults to <input_stem>.csv next to the input file.",
    )
    parser.add_argument(
        "--metrics-json",
        help="Path to output metrics JSON. Defaults to <input_stem>.metrics.json next to the input file.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=500,
        help="Max length for long text fields in the CSV. Default: 500",
    )
    return parser.parse_args()


def truncate_text(value: Any, max_len: int) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def json_dumps(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def extract_last_solution(text: Optional[str]) -> str:
    if not text:
        return ""
    matches = SOLUTION_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return ""


def extract_last_assistant_message(history: Iterable[Dict[str, Any]]) -> str:
    assistant_messages = [item.get("content", "") for item in history if item.get("role") == "assistant"]
    return assistant_messages[-1] if assistant_messages else ""


def build_row(record: Dict[str, Any], max_text_length: int) -> Dict[str, Any]:
    state = record.get("state", {}) or {}
    task = record.get("task", {}) or {}
    history = state.get("history", []) or []
    latest_output = state.get("latest_output", {}) or {}
    action_count = state.get("agent_action_count", {}) or {}
    token_counter = state.get("token_counter", {}) or {}
    metadata = task.get("metadata", {}) or {}

    final_assistant_message = extract_last_assistant_message(history)
    latest_content = latest_output.get("content", "")
    predicted_solution = extract_last_solution(final_assistant_message) or extract_last_solution(latest_content)

    row = {
        "task_id": task.get("task_id", ""),
        "task_name": task.get("task_name", ""),
        "prompt": truncate_text(task.get("prompt", ""), max_text_length),
        "reference": task.get("reference", ""),
        "success": int(bool(state.get("success", False))),
        "finished": int(bool(state.get("finished", False))),
        "terminate_reason": state.get("terminate_reason", ""),
        "error": truncate_text(state.get("error", ""), max_text_length),
        "history_messages": len(history),
        "assistant_turns": sum(1 for item in history if item.get("role") == "assistant"),
        "user_turns": sum(1 for item in history if item.get("role") == "user"),
        "predicted_solution": predicted_solution,
        "latest_observation": truncate_text(latest_output.get("observation", ""), max_text_length),
        "latest_feedback": truncate_text(latest_output.get("feedback", ""), max_text_length),
        "latest_feedback_type": latest_output.get("feedback_type", ""),
        "latest_output_success": latest_output.get("success", ""),
        "latest_content": truncate_text(latest_content, max_text_length),
        "propose_solution_count": action_count.get("propose_solution", 0),
        "use_tool_count": action_count.get("use_tool", 0),
        "invalid_action_count": action_count.get("invalid_action", 0),
        "prompt_tokens": token_counter.get("prompt_tokens", 0),
        "completion_tokens": token_counter.get("completion_tokens", 0),
        "total_tokens": token_counter.get("total_tokens", 0),
        "metadata_json": json_dumps(metadata),
    }
    return row


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "total_examples": 0,
            "accuracy": 0.0,
            "finished_rate": 0.0,
            "with_prediction_rate": 0.0,
            "avg_assistant_turns": 0.0,
            "avg_tool_uses": 0.0,
            "avg_invalid_actions": 0.0,
            "avg_total_tokens": 0.0,
            "terminate_reason_breakdown": {},
        }

    terminate_reason_counter = Counter(row["terminate_reason"] or "UNKNOWN" for row in rows)
    success_values = [row["success"] for row in rows]
    finished_values = [row["finished"] for row in rows]
    prediction_values = [1 if row["predicted_solution"] else 0 for row in rows]

    metrics = {
        "total_examples": total,
        "correct_examples": int(sum(success_values)),
        "accuracy": round(sum(success_values) / total, 6),
        "finished_rate": round(sum(finished_values) / total, 6),
        "with_prediction_rate": round(sum(prediction_values) / total, 6),
        "avg_assistant_turns": round(safe_mean([row["assistant_turns"] for row in rows]), 4),
        "avg_tool_uses": round(safe_mean([row["use_tool_count"] for row in rows]), 4),
        "avg_invalid_actions": round(safe_mean([row["invalid_action_count"] for row in rows]), 4),
        "avg_total_tokens": round(safe_mean([row["total_tokens"] for row in rows]), 2),
        "avg_prompt_tokens": round(safe_mean([row["prompt_tokens"] for row in rows]), 2),
        "avg_completion_tokens": round(safe_mean([row["completion_tokens"] for row in rows]), 2),
        "terminate_reason_breakdown": dict(sorted(terminate_reason_counter.items())),
    }
    return metrics


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_records(input_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {input_path}: {exc}") from exc
    return records


def print_summary(metrics: Dict[str, Any], output_csv: Path, metrics_json: Path) -> None:
    print(f"CSV written to: {output_csv}")
    print(f"Metrics written to: {metrics_json}")
    print("")
    print("Summary")
    print(f"  total_examples      : {metrics['total_examples']}")
    print(f"  correct_examples    : {metrics.get('correct_examples', 0)}")
    print(f"  accuracy            : {metrics['accuracy']:.4%}")
    print(f"  finished_rate       : {metrics['finished_rate']:.4%}")
    print(f"  with_prediction_rate: {metrics['with_prediction_rate']:.4%}")
    print(f"  avg_assistant_turns : {metrics['avg_assistant_turns']}")
    print(f"  avg_tool_uses       : {metrics['avg_tool_uses']}")
    print(f"  avg_invalid_actions : {metrics['avg_invalid_actions']}")
    print(f"  avg_total_tokens    : {metrics['avg_total_tokens']}")
    print("  terminate_reasons   :")
    for reason, count in metrics["terminate_reason_breakdown"].items():
        print(f"    - {reason}: {count}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else input_path.with_suffix(".csv")
    metrics_json = (
        Path(args.metrics_json).expanduser().resolve()
        if args.metrics_json
        else input_path.with_suffix(".metrics.json")
    )

    records = load_records(input_path)
    rows = [build_row(record, args.max_text_length) for record in records]
    metrics = compute_metrics(rows)

    write_csv(rows, output_csv)
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print_summary(metrics, output_csv, metrics_json)


if __name__ == "__main__":
    main()
