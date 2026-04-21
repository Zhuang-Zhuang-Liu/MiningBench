import argparse
import csv
import json
import math
import re
from pathlib import Path


SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.S)
RULE_RE = re.compile(r"<answer_rule>(.*?)</answer_rule>", re.S)
HELPERS = {
    "abs": abs,
    "log": math.log,
    "log1p": math.log1p,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "max": max,
    "min": min,
}


def extract_answer(text):
    if not text:
        return None
    m = SOLUTION_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def extract_rule(text):
    text = extract_answer(text)
    if not text:
        return None
    m = RULE_RE.search(text)
    return m.group(1).strip() if m else None


def normalize_rule(rule, task_type):
    if not rule:
        return None
    rule = rule.strip().replace("^", "**")
    if task_type == "classification" and " if " not in rule and "target" not in rule:
        return f"target = 1 if {rule} else 0"
    return rule


def load_prediction(line):
    obj = json.loads(line)
    if "state" in obj and "task" in obj:
        state = obj["state"]
        history = state.get("history", [])
        answer = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                answer = extract_answer(msg.get("content", ""))
                if answer:
                    break
        actions = state.get("agent_action_count", {})
        tokens = state.get("token_counter", {})
        return {
            "task_id": obj["task"]["task_id"],
            "output_answer": answer,
            "original_success": state.get("success"),
            "terminate_reason": state.get("terminate_reason"),
            "error": state.get("error"),
            "dialogue_turns": len(history) // 2,
            "prompt_tokens": tokens.get("prompt_tokens"),
            "completion_tokens": tokens.get("completion_tokens"),
            "total_tokens": tokens.get("total_tokens"),
            "tool_turns": actions.get("use_tool", 0),
            "propose_solution_turns": actions.get("propose_solution", 0),
            "invalid_action_turns": actions.get("invalid_action", 0),
        }
    return {"task_id": obj.get("task_id"), "output_answer": obj.get("output_answer")}


def load_manifest(path):
    tasks = {}
    root = path.parent.parent
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tasks[obj["id"]] = {
                "task_type": obj["ground_truth"].get("task_type", "classification"),
                "data_path": root / obj["input_data_path"],
                "expected_rule": obj["ground_truth"]["answer_rule"],
            }
    return tasks


def eval_classification(rule, rows):
    expr = rule.split(" if ", 1)[1].rsplit(" else ", 1)[0]
    mismatches = 0
    for row in rows:
        env = {k: float(v) for k, v in row.items() if k != "target"}
        env.update(HELPERS)
        pred = int(bool(eval(expr, {"__builtins__": {}}, env)))
        mismatches += pred != int(row["target"])
    total = len(rows)
    return {"mismatches": mismatches, "accuracy": 1 - mismatches / total, "explains_all": mismatches == 0}


def eval_regression(rule, rows, tol):
    expr = rule.split("=", 1)[1].strip()
    errs = []
    for row in rows:
        env = {k: float(v) for k, v in row.items() if k != "target"}
        env.update(HELPERS)
        pred = float(eval(expr, {"__builtins__": {}}, env))
        errs.append((pred - float(row["target"])) ** 2)
    rmse = math.sqrt(sum(errs) / len(errs))
    return {"rmse": rmse, "explains_all": rmse <= tol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="raw results.jsonl")
    parser.add_argument("--benchmark", type=Path, default=Path("/Users/jinming.liu/Documents/base/MiningEval/MiningBench/MiningBench-v1/benchmark.jsonl"))
    parser.add_argument("--rmse-tol", type=float, default=1e-9)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.benchmark)
    output_path = args.output or args.input.with_suffix(".rule_eval.jsonl")
    total_count = 0
    original_success_count = 0
    explains_all_count = 0
    tool_use_count = 0
    executable_count = 0
    original_false_but_semantic_true_count = 0
    prompt_tokens_sum = 0
    prompt_tokens_count = 0
    completion_tokens_sum = 0
    completion_tokens_count = 0
    total_tokens_sum = 0
    total_tokens_count = 0
    dialogue_turns_sum = 0
    dialogue_turns_count = 0
    tool_turns_sum = 0
    tool_turns_count = 0
    accuracy_sum = 0
    accuracy_count = 0

    with args.input.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            pred = load_prediction(line)
            task = manifest.get(pred["task_id"])
            row = {
                "task_id": pred["task_id"],
                "original_success": pred.get("original_success"),
                "terminate_reason": pred.get("terminate_reason"),
                "runtime_error": pred.get("error"),
                "dialogue_turns": pred.get("dialogue_turns"),
                "prompt_tokens": pred.get("prompt_tokens"),
                "completion_tokens": pred.get("completion_tokens"),
                "total_tokens": pred.get("total_tokens"),
                "tool_turns": pred.get("tool_turns"),
                "propose_solution_turns": pred.get("propose_solution_turns"),
                "invalid_action_turns": pred.get("invalid_action_turns"),
                "output_answer": pred["output_answer"],
                "answer_rule": extract_rule(pred["output_answer"]),
            }
            if not task:
                row.update({"executable": False, "error": "missing_task"})
            else:
                row["task_type"] = task["task_type"]
                row["expected_rule"] = task["expected_rule"]
                row["normalized_answer_rule"] = normalize_rule(row["answer_rule"], task["task_type"])
            if not task or not row.get("normalized_answer_rule"):
                row.update({"executable": False, "error": "missing_task_or_rule"})
            else:
                try:
                    with task["data_path"].open("r", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    row["executable"] = True
                    row.update(
                        eval_regression(row["normalized_answer_rule"], rows, args.rmse_tol)
                        if task["task_type"] == "regression"
                        else eval_classification(row["normalized_answer_rule"], rows)
                    )
                except Exception as e:
                    row.update({"executable": False, "error": f"{type(e).__name__}: {e}"})
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_count += 1
            original_success_count += bool(row.get("original_success"))
            explains_all_count += bool(row.get("explains_all"))
            tool_use_count += (row.get("tool_turns", 0) or 0) > 0
            executable_count += bool(row.get("executable"))
            original_false_but_semantic_true_count += (not bool(row.get("original_success"))) and bool(row.get("explains_all"))
            if row.get("prompt_tokens") is not None:
                prompt_tokens_sum += row["prompt_tokens"]
                prompt_tokens_count += 1
            if row.get("completion_tokens") is not None:
                completion_tokens_sum += row["completion_tokens"]
                completion_tokens_count += 1
            if row.get("total_tokens") is not None:
                total_tokens_sum += row["total_tokens"]
                total_tokens_count += 1
            if row.get("dialogue_turns") is not None:
                dialogue_turns_sum += row["dialogue_turns"]
                dialogue_turns_count += 1
            if row.get("tool_turns") is not None:
                tool_turns_sum += row["tool_turns"]
                tool_turns_count += 1
            if row.get("accuracy") is not None and row.get("executable"):
                accuracy_sum += row["accuracy"]
                accuracy_count += 1

    print(f"wrote jsonl to {output_path}")
    if total_count:
        print(f"original_success_rate: {original_success_count / total_count:.4f}")
        print(f"semantic_success_rate: {explains_all_count / total_count:.4f}")
        if prompt_tokens_count:
            print(f"avg_prompt_tokens: {prompt_tokens_sum / prompt_tokens_count:.2f}")
        if completion_tokens_count:
            print(f"avg_completion_tokens: {completion_tokens_sum / completion_tokens_count:.2f}")
        if total_tokens_count:
            print(f"avg_total_tokens: {total_tokens_sum / total_tokens_count:.2f}")
        if dialogue_turns_count:
            print(f"avg_dialogue_turns: {dialogue_turns_sum / dialogue_turns_count:.2f}")
        if tool_turns_count:
            print(f"avg_tool_turns: {tool_turns_sum / tool_turns_count:.2f}")
        print(f"tool_use_rate: {tool_use_count / total_count:.4f}")
        print(f"executable_rate: {executable_count / total_count:.4f}")
        print(f"zero_mismatch_count: {explains_all_count}")
        print(f"zero_mismatch_rate: {explains_all_count / total_count:.4f}")
        if accuracy_count:
            print(f"avg_accuracy_on_executable: {accuracy_sum / accuracy_count:.4f}")
        print(f"original_false_but_semantic_true_count: {original_false_but_semantic_true_count}")


if __name__ == "__main__":
    main()
