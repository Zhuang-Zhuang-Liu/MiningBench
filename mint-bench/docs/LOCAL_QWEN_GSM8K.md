# Local Qwen GSM8K Evaluation

This walkthrough follows the framework's standard flow:

1. register the evaluated model in `mint/configs/config_variables.py`
2. generate formal experiment configs with `mint/configs/generate_config.py`
3. run evaluation with `mint/main.py` using the generated config
4. inspect `data/outputs/.../results.jsonl`

## 1. Confirm the local serving endpoint

The local `LM Studio` server must expose an OpenAI-compatible endpoint:

```bash
curl http://localhost:1234/v1/models
```

At the time of writing, the available model id includes:

```text
qwen3.5-2b
```

This id must match the `model_name` in `mint/configs/config_variables.py`.

## 2. Environment setup

From the repository root:

```bash
cd mint-bench
conda activate mint
pip install -e .
```

This repository uses the pre-1.0 OpenAI Python SDK API style such as `openai.ChatCompletion.create(...)`.
The dependency file now pins `openai<1` so that the framework runs with its original agent implementation.

If the environment was not created yet:

```bash
conda env create -f environment.yml
conda activate mint
pip install -e .
```

If you already created the environment before this pin was added, align it manually:

```bash
cd mint-bench
conda activate mint
pip install 'openai<1'
pip install -e .
```

## 3. Generate framework configs

Run the framework's config generator:

```bash
cd mint-bench
python mint/configs/generate_config.py
```

This will generate configs under:

```text
configs/qwen3.5-2b/
```

For GSM8K, the generated configs are:

```text
configs/qwen3.5-2b/F=None/max1_p1+tool+cd/reasoning/gsm8k.json
configs/qwen3.5-2b/F=None/max2_p2+tool+cd/reasoning/gsm8k.json
configs/qwen3.5-2b/F=None/max3_p2+tool+cd/reasoning/gsm8k.json
configs/qwen3.5-2b/F=None/max4_p2+tool+cd/reasoning/gsm8k.json
configs/qwen3.5-2b/F=None/max5_p2+tool+cd/reasoning/gsm8k.json
```

## 4. Run GSM8K evaluation

If you want to learn the framework step by step, start with a single generated config:

```bash
cd mint-bench
python mint/main.py --exp_config configs/qwen3.5-2b/F=None/max5_p2+tool+cd/reasoning/gsm8k.json
```

If you only want a quick debug run over a few examples:

```bash
cd mint-bench
python mint/main.py --debug --exp_config configs/qwen3.5-2b/F=None/max5_p2+tool+cd/reasoning/gsm8k.json
```

## 5. Output locations

The result file will be written to:

```text
data/outputs/qwen3.5-2b/F=None/max5_p2+tool+cd/reasoning/gsm8k/results.jsonl
```

Logs will be appended to:

```text
data/outputs/qwen3.5-2b/F=None/max5_p2+tool+cd/reasoning/gsm8k/output.txt
```

## 6. Notes

- This repository uses `VLLMAgent` for open-source models served through an OpenAI-compatible API. In your local setup, `LM Studio` is being used as that compatible backend.
- `FEEDBACK_PROVIDER_LIST` is currently `None` only, so this flow evaluates task solving without the extra feedback-provider stage.
- `gsm8k` in this benchmark is under `data/processed/gsm8k/test_prompts.json`.
