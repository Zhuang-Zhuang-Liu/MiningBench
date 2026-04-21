# MiningBench: An Interactive Data Mining Evaluation Benchmark for Multi-turn Reasoning and Tool Usage

MiningBench is a benchmark that evaluates large language model (LLM) agents on open-ended, multi-step data mining tasks — specifically rule/factor discovery. Models must autonomously write and execute Python code across multiple turns to uncover hidden rules from tabular data, mimicking real-world data science workflows in quantitative finance and financial risk control.

## Overview

### Why Data Mining?

Existing agentic benchmarks focus on well-defined tasks (math, code generation, question answering). Data mining is fundamentally different:

- **No single solution path** — requires heuristic, exploratory reasoning
- **Unplanned difficulty** — task hardness is unknown at the start; the model must adapt based on feedback
- **High-dimensional action space** — correlation analysis, tree models, SHAP, feature engineering, etc.
- **Sparse intermediate signals** — unlike math proofs, partial solutions rarely give clear correctness cues
- **Noisy real-world data** — ground-truth rules are obscured by noise

## Task Design: Rule Discovery

Each task presents the model with a CSV file containing N feature columns and one target column. The model must identify the single hidden rule that generated target.

Rules span 20 categories across 6 families:

| Family | Rules | Difficulty |
|--------|-------|------------|
| Linear Weighted | Single-var threshold, dual-var linear, triple-var linear | easy–medium |
| Univariate Nonlinear | Cubic, square, sqrt, absolute deviation, log | easy–medium |
| Relative / Ratio | Log-ratio, difference, squared ratio | easy–hard |
| Interaction & Combination | Product, linear+interaction, quadratic cross | medium–hard |
| Threshold & Interval | Center deviation, max-cut, single-sided interval, double-sided interval | medium–hard |
| Conditional Gate | AND gate, OR gate | hard |

## Dataset Scale

The current public/reproducible version contains 200 anonymous tasks:

| Split | Count |
|-------|-------|
| Anonymous (feat_1, feat_2, …) | 200 |
| **Total** | **200** |

- Easy: 40 tasks · Medium: 70 · Hard: 90
- Classification tasks: 170 · Regression tasks: 30

## Repository Structure

```
MiningBench/
├── MiningBench/                  # Benchmark generator & raw data
│   ├── MiningBench-v1/           # Generated task data (CSV files + JSONL)
│   ├── MiningBench-v1-task_generator.py
│   └── ...
├── mint-bench/                   # Evaluation framework (built on MINT)
│   ├── mint/                     # Core framework code
│   │   ├── agents/               # LLM agent implementations
│   │   ├── configs/              # Config registry & generator
│   │   ├── tasks/                # Task definitions
│   │   └── main.py
│   ├── scripts/                  # Evaluation & analysis scripts
│   ├── configs/                  # Generated experiment config files
│   └── data/
│       ├── processed/            # Formatted task prompts
│       └── outputs/              # Model rollout results
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/Zhuang-Zhuang-Liu/MiningBench.git
cd MiningBench/mint-bench

# Option A: conda (recommended)
conda env create -f environment.yml
conda activate mint
pip install -e .

# Option B: Docker
./scripts/docker/run_interactive.sh
```

### 2. Prepare Benchmark Data

Convert MiningBench tasks to the MINT evaluation format:

```bash
python3 scripts/miningbench_to_mint_json.py \
  --input ../MiningBench/MiningBench-v1/benchmark.jsonl \
  --output data/processed/mining-bench/test_prompts.json \
  --data-root ../MiningBench/MiningBench-v1
```

### 3. Generate Experiment Configs

```bash
cd mint-bench
PYTHONPATH=$(pwd) python3 mint/configs/generate_config.py
```

This generates config files under `configs/<model_name>/...`.

### 4. Run Evaluation

```bash
cd mint-bench

# Local model via LM Studio (OpenAI-compatible)
PYTHONPATH=$(pwd) python3 mint/main.py \
  --exp_config configs/qwen/qwen3.5-9b/F=None/max20_p1+tool+cd/reasoning_da/mining-bench.json

# Multi-worker parallel evaluation (faster)
PYTHONPATH=$(pwd) python3 mint/main.py \
  --exp_config configs/qwen/qwen3.5-9b/F=None/max20_p1+tool+cd/reasoning_da/mining-bench.json \
  --num_workers 4

# Debug mode (single sample)
PYTHONPATH=$(pwd) python3 mint/main.py --debug \
  --exp_config configs/<model_name>/.../<task>.json
```

### 5. Score Results

Rule-based evaluation (aggregate benchmark scoring):

```bash
cd mint-bench
python3 scripts/eval_mining_bench.py data/outputs
```

Output: `data/outputs/<model_name>/results.merged.rule_eval.jsonl` and `.md`

## Evaluation Metrics

### Outcome Metrics

| Metric | Description |
|--------|-------------|
| Success Rate | Fraction of samples whose predicted rule is correct |
| Rule Match | Exact match of the discovered rule |

### Process Metrics

| Metric | Description |
|--------|-------------|
| Avg. Turns | Average number of interaction turns |
| Tool Usage | Frequency and correctness of tool calls |
| Code Execution | Success rate of Python code execution |

## Citation

If you use MiningBench in your research, please cite:

```bibtex
@misc{miningbench2026,
  title={MiningBench: An Interactive Data Mining Evaluation Benchmark for Multi-turn Reasoning and Tool Usage},
  author={Liu, Zhuang-Zhuang},
  year={2026},
  url={https://github.com/Zhuang-Zhuang-Liu/MiningBench}
}
```

## License

MIT License

## Acknowledgments

This benchmark is built on top of the [MINT](https://github.com/xingyaoww/mint-bench) framework.
