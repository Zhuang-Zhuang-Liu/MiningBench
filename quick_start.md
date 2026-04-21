### quick start

# data process
python3 data-process/miningbench_to_mint_json.py \
  --input MiningBench/MiningBench-v1/benchmark.jsonl \
  --output mint-bench/data/processed/mining-bench/test_prompts.json \
  --data-root MiningBench/MiningBench-v1

# run config
cd mint-bench
PYTHONPATH=$(pwd) python3 mint/configs/generate_config.py


# rollout with qwen3.5-9b by lm_studio
cd mint-bench
PYTHONPATH=$(pwd) ALFWORLD_DATA=data/processed/alfworld \
./.venv/bin/python mint/main.py \
--exp_config configs/qwen/qwen3.5-9b/F=None/max20_p1+tool+cd/reasoning_da/mining-bench.json


# rollout with glm4.7 by api
cd mint-bench
PYTHONPATH=$(pwd) ALFWORLD_DATA=data/processed/alfworld \
./.venv/bin/python mint/main.py \
--exp_config configs/glm-4.7/F=None/max20_p1+tool+cd/reasoning_da/mining-bench.json


# eval 
cd mint-bench
./.venv/bin/python  scripts/eval_mining_bench.py \
'data/outputs/qwen3.5-9b/F=None/max20_p1+tool+cd/reasoning_da/mining-bench/results.jsonl'


