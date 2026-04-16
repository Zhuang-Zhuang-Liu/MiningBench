# MiningBench: An Interactive Data Mining Evaluation Benchmark for Multi-turn Reasoning and Tool Usage

**MiningBench** is a benchmark that evaluates large language model (LLM) agents on open-ended, multi-step data mining tasks — specifically *rule/factor discovery*. Models must autonomously write and execute Python code across multiple turns to uncover hidden rules from tabular data, mimicking real-world data science workflows in quantitative finance and financial risk control.

---

## Overview

### Why Data Mining?

Existing agentic benchmarks focus on well-defined tasks (math, code generation, question answering). Data mining is fundamentally different:

- **No single solution path** — requires heuristic, exploratory reasoning
- **Unplanned difficulty** — task hardness is unknown at the start; the model must adapt based on feedback
- **High-dimensional action space** — correlation analysis, tree models, SHAP, feature engineering, etc.
- **Sparse intermediate signals** — unlike math proofs, partial solutions rarely give clear correctness cues
- **Noisy real-world data** — ground-truth rules are obscured by noise

### Task Design: Rule Discovery

Each task presents the model with a CSV file containing `N` feature columns and one `target` column. The model must identify the single hidden rule that generated `target`.

Rules span **20 categories** across 6 families:

| Family | Rules | Difficulty |
|---|---|---|
| Linear Weighted | Single-var threshold, dual-var linear, triple-var linear | easy–medium |
| Univariate Nonlinear | Cubic, square, sqrt, absolute deviation, log | easy–medium |
| Relative / Ratio | Log-ratio, difference, squared ratio | easy–hard |
| Interaction & Combination | Product, linear+interaction, quadratic cross | medium–hard |
| Threshold & Interval | Center deviation, max-cut, single-sided interval, double-sided interval | medium–hard |
| Conditional Gate | AND gate, OR gate | hard |

### Dataset Scale

The current public/reproducible version contains **200 anonymous tasks**:

| Split | Count |
|---|---|
| Anonymous (feat_1, feat_2, …) | 200 |
| **Total** | **200** |

- **Easy**: 40 tasks · **Medium**: 70 · **Hard**: 90
- **Classification** tasks: 170 · **Regression** tasks: 30

---

## Repository Structure

```
MiningEval/
├── MiningBench/                  # Benchmark generator & raw data
│   ├── MiningBench-v1/           # Generated task data (CSV files + JSONL)
│   ├── MiningBench-v1-task_generator.py
│   └── ...
├── data-process/
│   └── miningbench_to_mint_json.py
├── mint-bench/                   # Evaluation framework (built on MINT)
│   ├── mint/                     # Core framework code
│   │   ├── agents/               # LLM agent implementations
│   │   ├── configs/              # Config registry & generator
│   │   ├── tasks/                # Task definitions
│   │   └── main.py
│   ├── rubric/                   # Trajectory quality rubric evaluator
│   │   ├── score_trajectory_rubric.py
│   │   └── trajectory_rule_discovery_rubric.jsonl
│   ├── scripts/                  # Evaluation & analysis scripts
│   ├── configs/                  # Generated experiment config files
│   └── data/
│       ├── processed/            # Formatted task prompts
│       └── outputs/              # Model rollout results
└── paper/                        # Research paper source
```

---

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
python3 data-process/miningbench_to_mint_json.py \
  --input MiningBench/MiningBench-v1/benchmark.jsonl \
  --output mint-bench/data/processed/mining-bench/test_prompts.json \
  --data-root MiningBench/MiningBench-v1
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

# Remote API model (e.g. via OpenRouter)
PYTHONPATH=$(pwd) python3 mint/main.py \
  --exp_config configs/nvidia/nemotron-3-super-120b-a12b:free/F=None/max20_p1+tool+cd/reasoning_da/mining-bench.json

# Debug mode (single sample)
PYTHONPATH=$(pwd) python3 mint/main.py --debug \
  --exp_config configs/<model_name>/.../<task>.json
```

### 5. Score Results

**Rule-based evaluation** (aggregate benchmark scoring):

```bash
cd mint-bench
python3 scripts/eval_mining_bench.py data/outputs
```

Output: `data/outputs/<model_name>/results.merged.rule_eval.jsonl` and `.md`

**Trajectory rubric evaluation** (LLM-as-judge, 10 binary dimensions):

```bash
export MININGEVAL_API_KEY='YOUR_API_KEY'
python3 mint-bench/rubric/score_trajectory_rubric.py \
  mint-bench/data/outputs/<model_name>/results.merged.jsonl \
  --mode online
```

---

## Evaluation Metrics

### Outcome Metrics

| Metric | Description |
|---|---|
| Success Rate | Fraction of samples whose predicted rule is semantically correct under the evaluator |
| Feature Set Accuracy | Whether the model identified the exact set of correct feature variables |
| Near Miss Rate | Fraction of non-perfect executable samples that still achieve very high accuracy / very low RMSE |

### Trajectory Rubrics (binary 0 / 1 scale)

Ten dimensions evaluate *how* the model explored:

1. **Candidate Space Reduction** — Does the model efficiently narrow down variables early?
2. **Representation Exploration** — Does the model test diverse functional forms (linear, nonlinear, interaction)?
3. **Evidence-Based Hypothesis Pivoting** — Does the model switch direction based on data signals rather than randomly?
4. **Metric-Anchored Key Decisions** — Are accept/reject decisions backed by concrete metrics?
5. **Computational Tool Leverage** — Does the model use statistical tools, ML models, or enumeration effectively?
6. **Post-Fit Mechanism Verification** — After finding a high-fit formula, does the model verify it is the true mechanism?
7. **Rule Expression Normalization** — Is the final rule compact, reproducible, and clearly expressed?
8. **Submission-Evidence Consistency** — Does the final answer align with intermediate reasoning?
9. **Stopping Quality** — Does the model stop once the rule is sufficiently stabilized?
10. **Budget Hygiene** — Does the model avoid repeated invalid or wasteful actions?

---

## Adding a New Model

Edit `mint-bench/mint/configs/config_variables.py` and add an entry to `EVALUATED_MODEL_LIST`:

```python
{
    "agent_class": "VLLMAgent",   # for local OpenAI-compatible servers
    "config": {
        "model_name": "your-model-id",
        "chat_mode": True,
        "max_tokens": 512,
        "temperature": 0.0,
        "openai.api_base": "http://localhost:1234/v1",
        "add_system_message": False,
    },
},
```

Then regenerate configs and run:

```bash
PYTHONPATH=$(pwd) python3 mint/configs/generate_config.py
PYTHONPATH=$(pwd) python3 mint/main.py --exp_config configs/<model_name>/.../<task>.json
```

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for detailed instructions on adding models, tasks, and handling optional dependencies.

---

## Output Format

Model outputs are stored as JSONL files:

```
data/outputs/<model_name>/F=None/max20_p1+tool+cd/reasoning_da/mining-bench/
├── results.jsonl              # raw trajectories
├── results.merged.jsonl       # merged multi-run results
├── results.merged.rule_eval.jsonl   # rule accuracy scores
├── results.merged.rule_eval.md      # human-readable summary
└── output.txt                 # execution log
```

Each task in `MiningBench` follows this JSONL schema:

```json
{
  "id": "nb_v1_task_0001",
  "split": "anonymous",
  "difficulty": "medium",
  "input_data_path": "task_01/data.csv",
  "ground_truth": {
    "rule_name": "双变量线性与交互混合规则",
    "task_type": "classification",
    "answer_features": ["feat_1", "feat_4"],
    "answer_rule": "target = 1 if (2*feat_1 + -3*feat_4 + feat_1*feat_4) > 7 else 0"
  }
}
```

---

## Benchmark Generation

To regenerate the benchmark from scratch:

```bash
python3 MiningBench/MiningBench-v1-task_generator.py --validate
```

The current generator/build pipeline creates the 200-task anonymous benchmark by:
1. Sampling random multivariate data
2. Selecting one of 20 rule templates as ground truth
3. Injecting the rule's effect into the target column
4. Optionally adding noise

---



## 🤝 Main contributors ( welcome to join us! )
<table border="0">
  <tbody>
    <tr align="center">
      <td width="130">
        <a href="https://github.com/Zhuang-Zhuang-Liu"><img width="70" height="70" src="https://github.com/Zhuang-Zhuang-Liu.png?s=40" alt="pic"></a><br>
        <a href="https://github.com/Zhuang-Zhuang-Liu">ZhuangZhuangLiu</a>
        <p> We Lab </p>
      </td>
      <td width="150">
        <a href="https://github.com/jd-SearchEngines"><img width="70" height="70" src="https://github.com/jd-SearchEngines.png?s=40" alt="pic"></a><br>
        <a href="https://github.com/jd-SearchEngines">jd-SearchEngines</a>
        <p> Shopee </p>
      </td>
    </tr>
  </tbody>
</table>



## Citation

If you use MiningBench in your research, please cite:

```bibtex
@misc{miningbench2026,
  title   = {MiningBench: An Interactive Data Mining Evaluation Benchmark for Multi-turn Reasoning and Tool Usage},
  author  = {Zhuangzhuang Liu and Jingdong Deng},
  year    = {2026},
  note    = {https://github.com/Zhuang-Zhuang-Liu/MiningBench}
}
```

This project builds on the [MINT](https://arxiv.org/abs/2309.10691) evaluation framework:

```bibtex
@misc{wang2023mint,
  title   = {MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback},
  author  = {Xingyao Wang and Zihan Wang and Jiateng Liu and Yangyi Chen and Lifan Yuan and Hao Peng and Heng Ji},
  year    = {2023},
  eprint  = {2309.10691},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## License




This project is licensed under the MIT License. See [mint-bench/LICENSE](mint-bench/LICENSE) for details.
