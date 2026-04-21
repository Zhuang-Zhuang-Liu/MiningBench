#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


BENCHMARK = "nice_bench_rule_mining"
GENERATOR_VERSION = "v1.2.0"
QUESTION = """# 任务 {task_no:04d}
任务：基于提供的 `data.csv`（包含 {n_vars} 个特征列和 1 个 `target` 列），尽可能恢复生成 `target` 的底层全局规则
文件路径："{data_path}"
工具：你可以使用python来完成任务

# 特征说明
本任务中的特征名均为匿名脱敏字段，例如`feat_1`、`feat_2`等，不含任何业务语义，请不要根据字段名推断变量含义或规则形式。

# 任务提示：
1. 只存在1条统一的全局规则，不存在多条子规则。
2. 数据中可能存在一定的随机噪声。
3. 规则涉及的特征数量未知，应由数据本身决定；不要预设只依赖单个特征，也不要预设一定需要多个特征。
4. 允许任意形式的候选规则，最终答案应满足对 `target` 具有尽可能高的一致解释能力，且优先选择更简洁、可复现、可验证的表达。


# 输出格式：
最终答案只输出下面两个 XML 标签，不能包含任何额外解释、分析过程或其他文本。
`<answer_features>` 中填写规则实际使用到的全部特征名，按规则中出现顺序书写，使用英文逗号分隔。
    eg: <answer_features>feat_i, feat_j</answer_features>
`<answer_rule>` 中填写唯一规则表达式。
    eg: <answer_rule>target = 1 if a*feat_i ... > c else 0</answer_rule>
    eg: <answer_rule>target = log(feat_i) + c</answer_rule>
"""


@dataclass(frozen=True)
class RuleSpec:
    rule_id: int
    name: str
    difficulty: str
    category: str
    n_vars: int
    positive_vars: tuple[int, ...]
    builder: Callable[["TaskContext"], dict]


@dataclass
class TaskContext:
    rng: random.Random
    headers: list[str]
    rows: list[list[float]]

    def col(self, idx: int) -> list[float]:
        return [row[idx] for row in self.rows]

    def name(self, idx: int) -> str:
        return self.headers[idx]


def nz_int(rng: random.Random, low: int = -5, high: int = 5) -> int:
    return rng.choice([x for x in range(low, high + 1) if x != 0])


def q(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * p)))]


def rule_const(x: float | int) -> float:
    x = round(float(x), 1)
    return 0.0 if x == 0 else x


def fmt(x: float | int) -> str:
    if isinstance(x, int) or abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


def expr_sum(parts: list[str]) -> str:
    return " + ".join(parts).replace("+ -", "- ")


def expr_minus(lhs: str, rhs: float | int) -> str:
    return f"{lhs} + {fmt(abs(rhs))}" if rhs < 0 else f"{lhs} - {fmt(rhs)}"


def classify(signal: list[float], threshold: float, rng: random.Random, noise_ratio: float) -> list[int]:
    if noise_ratio <= 0:
        return [int(x > threshold) for x in signal]
    std = statistics.pstdev(signal) or 1.0
    return [int(x + rng.gauss(0, noise_ratio * std) > threshold) for x in signal]


def base_payload(ctx: TaskContext, feature_ids: list[int], signal: list[float], rule_expr: str, params: dict) -> dict:
    threshold = rule_const(params.get("threshold", q(signal, 0.5)))
    params["threshold"] = threshold
    return {
        "features": [ctx.name(i) for i in feature_ids],
        "signal": signal,
        "answer_rule": f"target = 1 if {rule_expr} > {fmt(threshold)} else 0",
        "parameters": params,
    }


def regression_payload(ctx: TaskContext, feature_ids: list[int], signal: list[float], rule_expr: str, params: dict | None = None) -> dict:
    return {
        "features": [ctx.name(i) for i in feature_ids],
        "signal": signal,
        "task_type": "regression",
        "answer_rule": f"target = {rule_expr}",
        "parameters": params or {},
    }


def r1(ctx: TaskContext) -> dict:
    i = 0
    direction = ctx.rng.choice([">", "<"])
    threshold = rule_const(q(ctx.col(i), 0.5))
    if direction == ">":
        signal = ctx.col(i)
        target_threshold = threshold
    else:
        signal = [-v for v in ctx.col(i)]
        target_threshold = rule_const(-threshold)
    f = ctx.name(i)
    return {
        "features": [f],
        "signal": signal,
        "answer_rule": f"target = 1 if {f} {direction} {fmt(threshold)} else 0",
        "parameters": {"direction": direction, "threshold": target_threshold},
    }


def r2(ctx: TaskContext) -> dict:
    b = [nz_int(ctx.rng), nz_int(ctx.rng)]
    sig = [b[0] * r[0] + b[1] * r[1] for r in ctx.rows]
    expr = expr_sum([f"{b[0]}*{ctx.name(0)}", f"{b[1]}*{ctx.name(1)}"])
    return base_payload(ctx, [0, 1], sig, expr, {"weights": {ctx.name(0): b[0], ctx.name(1): b[1]}})


def r3(ctx: TaskContext) -> dict:
    b = [nz_int(ctx.rng), nz_int(ctx.rng), nz_int(ctx.rng)]
    sig = [sum(b[j] * r[j] for j in range(3)) for r in ctx.rows]
    expr = expr_sum([f"{b[j]}*{ctx.name(j)}" for j in range(3)])
    return base_payload(ctx, [0, 1, 2], sig, expr, {"weights": {ctx.name(j): b[j] for j in range(3)}})


def r4(ctx: TaskContext) -> dict:
    f = ctx.name(0)
    intercept = rule_const(ctx.rng.uniform(-5.0, 5.0))
    sig = [v**3 + intercept for v in ctx.col(0)]
    return regression_payload(ctx, [0], sig, expr_sum([f"{f}**3", fmt(intercept)]), {"intercept": intercept})


def r5(ctx: TaskContext) -> dict:
    f = ctx.name(0)
    direction = ctx.rng.choice([">", "<"])
    values = [v**2 for v in ctx.col(0)]
    threshold = rule_const(q(values, 0.5))
    sig = values if direction == ">" else [-v for v in values]
    target_threshold = threshold if direction == ">" else rule_const(-threshold)
    return {
        "features": [f],
        "signal": sig,
        "answer_rule": f"target = 1 if {f}**2 {direction} {fmt(threshold)} else 0",
        "parameters": {"direction": direction, "threshold": target_threshold},
    }


def r6(ctx: TaskContext) -> dict:
    f = ctx.name(0)
    intercept = rule_const(ctx.rng.uniform(-5.0, 5.0))
    sig = [math.sqrt(v) + intercept for v in ctx.col(0)]
    return regression_payload(ctx, [0], sig, expr_sum([f"{f}**0.5", fmt(intercept)]), {"intercept": intercept})


def r7(ctx: TaskContext) -> dict:
    f = ctx.name(0)
    direction = ctx.rng.choice([">", "<"])
    values = [abs(v) for v in ctx.col(0)]
    threshold = rule_const(q(values, 0.5))
    sig = values if direction == ">" else [-v for v in values]
    target_threshold = threshold if direction == ">" else rule_const(-threshold)
    return {
        "features": [f],
        "signal": sig,
        "answer_rule": f"target = 1 if abs({f}) {direction} {fmt(threshold)} else 0",
        "parameters": {"direction": direction, "threshold": target_threshold},
    }


def r8(ctx: TaskContext) -> dict:
    f = ctx.name(0)
    intercept = rule_const(ctx.rng.uniform(-5.0, 5.0))
    sig = [math.log(v) + intercept for v in ctx.col(0)]
    return regression_payload(ctx, [0], sig, expr_sum([f"log({f})", fmt(intercept)]), {"intercept": intercept})


def r9(ctx: TaskContext) -> dict:
    direction = ctx.rng.choice([">", "<"])
    values = [math.log(r[0] / r[1]) for r in ctx.rows]
    threshold = rule_const(q(values, 0.5))
    sig = values if direction == ">" else [-v for v in values]
    target_threshold = threshold if direction == ">" else rule_const(-threshold)
    return {
        "features": [ctx.name(0), ctx.name(1)],
        "signal": sig,
        "answer_rule": f"target = 1 if log({ctx.name(0)} / {ctx.name(1)}) {direction} {fmt(threshold)} else 0",
        "parameters": {"direction": direction, "threshold": target_threshold},
    }


def r10(ctx: TaskContext) -> dict:
    direction = ctx.rng.choice([">", "<"])
    diffs = [r[0] - r[1] for r in ctx.rows]
    threshold = rule_const(q(diffs, 0.5))
    sig = diffs if direction == ">" else [-v for v in diffs]
    target_threshold = threshold if direction == ">" else rule_const(-threshold)
    expr = f"{ctx.name(0)} - {ctx.name(1)}"
    return {
        "features": [ctx.name(0), ctx.name(1)],
        "signal": sig,
        "answer_rule": f"target = 1 if {expr} {direction} {fmt(threshold)} else 0",
        "parameters": {"direction": direction, "threshold": target_threshold},
    }


def r11(ctx: TaskContext) -> dict:
    direction = ctx.rng.choice([">", "<"])
    values = [(r[0] / r[1]) ** 2 for r in ctx.rows]
    threshold = rule_const(q(values, 0.5))
    sig = values if direction == ">" else [-v for v in values]
    target_threshold = threshold if direction == ">" else rule_const(-threshold)
    return {
        "features": [ctx.name(0), ctx.name(1)],
        "signal": sig,
        "answer_rule": f"target = 1 if ({ctx.name(0)} / {ctx.name(1)})**2 {direction} {fmt(threshold)} else 0",
        "parameters": {"direction": direction, "threshold": target_threshold},
    }


def r12(ctx: TaskContext) -> dict:
    direction = ctx.rng.choice([">", "<"])
    products = [r[0] * r[1] for r in ctx.rows]
    threshold = rule_const(q(products, 0.5))
    sig = products if direction == ">" else [-v for v in products]
    target_threshold = threshold if direction == ">" else rule_const(-threshold)
    expr = f"{ctx.name(0)} * {ctx.name(1)}"
    return {
        "features": [ctx.name(0), ctx.name(1)],
        "signal": sig,
        "answer_rule": f"target = 1 if {expr} {direction} {fmt(threshold)} else 0",
        "parameters": {"direction": direction, "threshold": target_threshold},
    }


def r13(ctx: TaskContext) -> dict:
    b = [nz_int(ctx.rng, -3, 3), nz_int(ctx.rng, -3, 3)]
    sig = [b[0] * r[0] + b[1] * r[1] + r[0] * r[1] for r in ctx.rows]
    expr = expr_sum([f"{b[0]}*{ctx.name(0)}", f"{b[1]}*{ctx.name(1)}", f"{ctx.name(0)}*{ctx.name(1)}"])
    return base_payload(ctx, [0, 1], sig, expr, {"weights": {ctx.name(0): b[0], ctx.name(1): b[1]}, "interaction_weight": 1})


def r14(ctx: TaskContext) -> dict:
    b = [ctx.rng.choice([-2, -1, 1, 2]), nz_int(ctx.rng, -5, 5)]
    sig = [b[0] * r[0] ** 2 + b[1] * r[1] for r in ctx.rows]
    expr = expr_sum([f"{b[0]}*{ctx.name(0)}**2", f"{b[1]}*{ctx.name(1)}"])
    return base_payload(ctx, [0, 1], sig, expr, {"weights": {ctx.name(0): b[0], ctx.name(1): b[1]}})


def r15(ctx: TaskContext) -> dict:
    f = ctx.name(0)
    k = rule_const(q(ctx.col(0), 0.5))
    sig = [abs(v - k) for v in ctx.col(0)]
    return base_payload(ctx, [0], sig, f"abs({expr_minus(f, k)})", {"center": k, "threshold": rule_const(q(sig, 0.7))})


def r16(ctx: TaskContext) -> dict:
    b = [nz_int(ctx.rng), nz_int(ctx.rng), nz_int(ctx.rng)]
    sig = [max(b[j] * r[j] for j in range(3)) for r in ctx.rows]
    expr = f"max({', '.join(f'{b[j]}*{ctx.name(j)}' for j in range(3))})"
    return base_payload(ctx, [0, 1, 2], sig, expr, {"weights": {ctx.name(j): b[j] for j in range(3)}})


def r17(ctx: TaskContext) -> dict:
    f = ctx.name(0)
    sig = ctx.col(0)
    lo, hi = rule_const(q(sig, 0.35)), rule_const(q(sig, 0.65))
    return {"features": [f], "signal": sig, "interval": (lo, hi), "answer_rule": f"target = 1 if {fmt(lo)} < {f} < {fmt(hi)} else 0", "parameters": {"lower": lo, "upper": hi}}


def r18(ctx: TaskContext) -> dict:
    b = [nz_int(ctx.rng), nz_int(ctx.rng)]
    sig = [b[0] * r[0] + b[1] * r[1] for r in ctx.rows]
    lo, hi = rule_const(q(sig, 0.35)), rule_const(q(sig, 0.65))
    expr = expr_sum([f"{b[0]}*{ctx.name(0)}", f"{b[1]}*{ctx.name(1)}"])
    return {"features": [ctx.name(0), ctx.name(1)], "signal": sig, "interval": (lo, hi), "answer_rule": f"target = 1 if {fmt(lo)} < ({expr}) < {fmt(hi)} else 0", "parameters": {"weights": {ctx.name(0): b[0], ctx.name(1): b[1]}, "lower": lo, "upper": hi}}


def r19(ctx: TaskContext) -> dict:
    f1, f2 = ctx.name(0), ctx.name(1)
    direction = ctx.rng.choice([">", "<"])
    k = rule_const(q(ctx.col(1), 0.35))
    n = rule_const(q(ctx.col(0), 0.35))
    sig = ctx.col(0) if direction == ">" else [-v for v in ctx.col(0)]
    target_threshold = n if direction == ">" else rule_const(-n)
    return {"features": [f2, f1], "signal": sig, "gate_col": 1, "gate_k": k, "threshold": target_threshold, "gate_type": "and", "answer_rule": f"target = 1 if ({f2} > {fmt(k)}) and ({f1} {direction} {fmt(n)}) else 0", "parameters": {"direction": direction, "k": k, "threshold": target_threshold}}


def r20(ctx: TaskContext) -> dict:
    f1, f2 = ctx.name(0), ctx.name(1)
    direction = ctx.rng.choice([">", "<"])
    k = rule_const(q(ctx.col(0), 0.75))
    n = rule_const(q(ctx.col(1), 0.75))
    sig = ctx.col(1) if direction == ">" else [-v for v in ctx.col(1)]
    target_threshold = n if direction == ">" else rule_const(-n)
    return {"features": [f1, f2], "signal": sig, "gate_col": 0, "gate_k": k, "threshold": target_threshold, "gate_type": "or", "answer_rule": f"target = 1 if ({f1} > {fmt(k)}) or ({f2} {direction} {fmt(n)}) else 0", "parameters": {"direction": direction, "k": k, "threshold": target_threshold}}


RULES = [
    RuleSpec(1, "单变量线性阈值规则", "easy", "线性加权类", 1, (), r1),
    RuleSpec(2, "双变量线性组合阈值规则", "easy", "线性加权类", 2, (), r2),
    RuleSpec(3, "三变量线性组合阈值规则", "medium", "线性加权类", 3, (), r3),
    RuleSpec(4, "单变量三次放大规则", "medium", "单变量非线性变换类", 1, (), r4),
    RuleSpec(5, "单变量平方放大规则", "easy", "单变量非线性变换类", 1, (), r5),
    RuleSpec(6, "单变量平方根压缩规则", "medium", "单变量非线性变换类", 1, (0,), r6),
    RuleSpec(7, "单变量绝对偏离规则", "medium", "单变量非线性变换类", 1, (), r7),
    RuleSpec(8, "单变量对数压缩规则", "medium", "单变量非线性变换类", 1, (0,), r8),
    RuleSpec(9, "双变量对数比值规则", "hard", "相对关系与比例差异类", 2, (0, 1), r9),
    RuleSpec(10, "双变量差值规则", "easy", "相对关系与比例差异类", 2, (), r10),
    RuleSpec(11, "双变量平方比值规则", "hard", "相对关系与比例差异类", 2, (0, 1), r11),
    RuleSpec(12, "双变量交互乘积规则", "medium", "交互与组合效应类", 2, (), r12),
    RuleSpec(13, "双变量线性与交互混合规则", "hard", "交互与组合效应类", 2, (), r13),
    RuleSpec(14, "二次交叉组合规则", "hard", "交互与组合效应类", 2, (), r14),
    RuleSpec(15, "单变量中心偏离规则", "hard", "阈值、区间与截断类", 1, (), r15),
    RuleSpec(16, "多变量最大值截断规则", "hard", "阈值、区间与截断类", 3, (), r16),
    RuleSpec(17, "单变量双边区间规则", "medium", "阈值、区间与截断类", 1, (), r17),
    RuleSpec(18, "双变量线性组合双边区间规则", "hard", "阈值、区间与截断类", 2, (), r18),
    RuleSpec(19, "双变量AND门控", "hard", "条件门控类", 2, (), r19),
    RuleSpec(20, "双变量OR门控", "hard", "条件门控类", 2, (), r20),
]


def make_rows(rng: random.Random, n_rows: int, n_features: int, positive: tuple[int, ...]) -> list[list[float]]:
    rows = []
    means = [rng.uniform(-2.5, 2.5) for _ in range(n_features)]
    scales = [rng.uniform(1.0, 5.0) for _ in range(n_features)]
    for _ in range(n_rows):
        row = [round(rng.gauss(means[i], scales[i]), 6) for i in range(n_features)]
        for idx in positive:
            row[idx] = round(rng.uniform(0.25, 12.0), 6)
        rows.append(row)
    return rows


def make_targets(payload: dict, rows: list[list[float]], rng: random.Random, noise_ratio: float) -> list[float | int]:
    if payload.get("task_type") == "regression":
        if noise_ratio <= 0:
            return payload["signal"]
        std = statistics.pstdev(payload["signal"]) or 1.0
        return [x + rng.gauss(0, noise_ratio * std) for x in payload["signal"]]
    if "interval" in payload:
        lo, hi = payload["interval"]
        return [int(lo < x < hi) for x in payload["signal"]]
    if "gate_col" in payload:
        sig = payload["signal"]
        if payload["gate_type"] == "and":
            return [int(row[payload["gate_col"]] > payload["gate_k"] and sig[i] > payload["threshold"]) for i, row in enumerate(rows)]
        return [int(row[payload["gate_col"]] > payload["gate_k"] or sig[i] > payload["threshold"]) for i, row in enumerate(rows)]
    return classify(payload["signal"], payload["parameters"]["threshold"], rng, noise_ratio)


def write_csv(path: Path, headers: list[str], rows: list[list[float]], targets: list[float | int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers + ["target"])
        writer.writerows(row + [target] for row, target in zip(rows, targets))


def answer_mismatches(rule: str, rows: list[dict[str, str]]) -> int:
    expr = rule.split(" if ", 1)[1].rsplit(" else ", 1)[0]
    helpers = {"abs": abs, "log": math.log, "max": max, "min": min}
    mismatches = 0
    for row in rows:
        env = {k: float(v) for k, v in row.items() if k != "target"}
        env.update(helpers)
        pred = int(bool(eval(expr, {"__builtins__": {}}, env)))
        mismatches += pred != int(row["target"])
    return mismatches


def regression_rmse(rule: str, rows: list[dict[str, str]]) -> float:
    expr = rule.split("=", 1)[1].strip()
    helpers = {"abs": abs, "log": math.log, "max": max, "min": min}
    squared_errors = []
    for row in rows:
        env = {k: float(v) for k, v in row.items() if k != "target"}
        env.update(helpers)
        pred = float(eval(expr, {"__builtins__": {}}, env))
        squared_errors.append((pred - float(row["target"])) ** 2)
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def question_for(task_no: int, n_features: int, data_path: str) -> str:
    return QUESTION.format(
        task_no=task_no,
        n_vars=n_features,
        data_path=data_path,
    )


def generate(output_dir: Path, cases_per_rule: int, n_rows: int, n_features: int, seed: int, noise_ratio: float) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    manifest = []
    task_no = 0
    headers = [f"feat_{i}" for i in range(1, n_features + 1)]
    for rule in RULES:
        for case_idx in range(1, cases_per_rule + 1):
            task_no += 1
            task_seed = seed + rule.rule_id * 1000 + case_idx
            rng = random.Random(task_seed)
            chosen = rng.sample(range(n_features), rule.n_vars)
            ordered_headers = [headers[i] for i in chosen] + [h for h in headers if h not in {headers[i] for i in chosen}]
            rows = make_rows(rng, n_rows, n_features, rule.positive_vars)
            ctx = TaskContext(rng, ordered_headers, rows)
            payload = rule.builder(ctx)
            task_type = payload.get("task_type", "classification")
            targets = make_targets(payload, rows, rng, noise_ratio)
            task_dir = output_dir / f"task_{task_no:04d}"
            task_dir.mkdir()
            rel_data = f"{output_dir.name}/{task_dir.name}/data.csv"
            write_csv(task_dir / "data.csv", ordered_headers, rows, targets)
            question = question_for(task_no, n_features, rel_data)
            task = {
                "id": f"{task_no:04d}",
                "split": "validation/anonymous",
                "difficulty": rule.difficulty,
                "question": question,
                "input_data_path": rel_data,
                "allowed_tools": ["python"],
                "output_schema": {"answer_features_tag": "answer_features", "answer_rule_tag": "answer_rule"},
                "ground_truth": {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "rule_category": rule.category,
                    "task_type": task_type,
                    "answer_features": payload["features"],
                    "answer_rule": payload["answer_rule"],
                    "parameters": {"noise_ratio": noise_ratio},
                },
                "metadata": {
                    "n_rows": n_rows,
                    "n_features": n_features,
                    "task_seed": task_seed,
                    "generator_version": GENERATOR_VERSION,
                },
            }
            manifest.append(task)
    with (output_dir / "benchmark.jsonl").open("w", encoding="utf-8") as f:
        for task in manifest:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def validate(output_dir: Path, expected: int) -> None:
    with (output_dir / "benchmark.jsonl").open(encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    assert len(manifest) == expected, (len(manifest), expected)
    for task in manifest:
        task_dir = output_dir / Path(task["input_data_path"]).parent.name
        assert (task_dir / "data.csv").exists()
        assert "question" in task and task["question"].strip()
        with (task_dir / "data.csv").open(encoding="utf-8") as f:
            csv_rows = list(csv.reader(f))
        assert len(csv_rows) == task["metadata"]["n_rows"] + 1
        assert csv_rows[0][-1] == "target"
        task_type = task["ground_truth"].get("task_type", "classification")
        if task_type == "classification":
            assert set(row[-1] for row in csv_rows[1:]) <= {"0", "1"}
            positives = sum(row[-1] == "1" for row in csv_rows[1:])
            assert 0 < positives < task["metadata"]["n_rows"], task["id"]
        if task["ground_truth"]["parameters"].get("noise_ratio", 0.0) == 0.0:
            with (task_dir / "data.csv").open(encoding="utf-8") as f:
                dict_rows = list(csv.DictReader(f))
            if task_type == "regression":
                rmse = regression_rmse(task["ground_truth"]["answer_rule"], dict_rows)
                assert rmse < 1e-9, (task["id"], rmse)
            else:
                mismatches = answer_mismatches(task["ground_truth"]["answer_rule"], dict_rows)
                assert mismatches == 0, (task["id"], mismatches)
    rules = {}
    for task in manifest:
        rid = task["ground_truth"]["rule_id"]
        rules[rid] = rules.get(rid, 0) + 1
    assert rules == {i: expected // len(RULES) for i in range(1, len(RULES) + 1)}, rules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("MiningBench-v1"))
    parser.add_argument("--cases-per-rule", type=int, default=10)
    parser.add_argument("--n-rows", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--noise-ratio", type=float, default=0.0)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    generate(args.output_dir, args.cases_per_rule, args.n_rows, args.n_features, args.seed, args.noise_ratio)
    if args.validate:
        validate(args.output_dir, len(RULES) * args.cases_per_rule)
    print(f"Generated {len(RULES) * args.cases_per_rule} anonymous tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
