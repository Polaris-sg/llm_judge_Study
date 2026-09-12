# LLM Judge Reliability for APR Patch Correctness

This repository is a replication package for an empirical study on using large language models as judges for automated program repair (APR) patch correctness assessment. It contains the experimental dataset used in the study, the LLM-judge prompt/runner scripts, released model outputs, and the script for cluster-aware robustness analysis.

## Study Overview

APR tools can generate overfitting patches: patches that pass available tests but do not truly fix the underlying defect. This study evaluates whether LLM-as-a-Judge can distinguish correct patches from overfitting patches, and whether its judgments are affected by metadata leakage or additional evidence in the prompt.

The study considers four research questions:

- RQ1: What is the agreement between LLM Judge and human ground-truth labels under sanitized conditions?
- RQ2: Does exposing benchmark-identifying information to LLM Judge systematically shift its judgments?
- RQ3: Does the type and content of evidence included in the prompt systematically bias LLM Judge toward labelling overfitting patches as correct?
- RQ4: Do the natural language explanations produced by LLM Judge provide trustworthy reasoning, or do they contain hallucinated or unsupported claims?

## Repository Structure

```text
llm_judge_reliability(git)/
+-- README.md
|
+-- data/
|   +-- main_dataset_326.jsonl
|
+-- figures/
|   +--rq2_summary.svg
|   +--rq3_summary.svg 
+-- scripts/
|   +-- config.py
|   +-- prompt_builder.py
|   +-- run_llm_judge.py
|   +-- cluster_robustness.py
+-- results/
|   +-- main_dataset_326(gpt-4o)/
|   +-- main_dataset_326(deepseek-v3)/
|
+-- analysis/
```

## API Configuration

The runner reads API credentials from `scripts/config.py`. Keep this file as a local placeholder until you rerun API calls, then fill in your own credentials:

```python
OPENAI_API_KEY = "sk-..."
OPENAI_BASE_URL = "https://"
```


## Dataset

The package includes the experimental dataset used by the LLM-judge runs:

- `data/main_dataset_326.jsonl`

It contains 326 patches from 163 Defects4J bugs. Each bug contributes exactly one correct patch and one overfitting patch, enabling paired comparisons that control for bug-level confounds.

Key properties:

- Bugs: 163
- Patches: 326
- Correct patches: 163
- Overfitting patches: 163
- Projects: Chart, Closure, Lang, Math, and Time
- Dataset selection seed: 42

Each JSONL record contains fields such as:

- `patch_id`
- `pair_id`
- `role`
- `ground_truth`
- `meta_bug_id`
- `meta_project`
- `meta_tool`
- `buggy_functions`
- `patch_diff`
- `test_cases`
- `execution_traces`
- `bug_description`
- `coverage_summary`
- `developer_patch_diff`
- `has_independent_developer_ref`

The raw source dataset and dataset-construction scripts are not included in this trimmed release.

## Prompt Settings

`scripts/prompt_builder.py` defines all prompt settings used in the experiments.

| Group | Setting | Description |
| --- | --- | --- |
| RQ2 Leakage | S0 | Sanitized baseline with buggy code and patch diff only |
| RQ2 Leakage | S1 | S0 + bug identifier |
| RQ2 Leakage | S2 | S0 + project name |
| RQ2 Leakage | S3 | S0 + APR tool name |
| RQ2 Leakage | S4 | S0 + bug identifier + project name + APR tool name |
| RQ2 Leakage | S4R | S0 + randomly shuffled metadata |
| RQ2 Leakage | S4F | S0 + fake bug identifier plus real project/tool metadata |
| RQ2 Leakage | S5 | S0 + developer reference patch |
| RQ3 Evidence | E0 | Same as S0 |
| RQ3 Evidence | E1 | E0 + failing test code |
| RQ3 Evidence | E2 | E0 + all-tests-passed signal |
| RQ3 Evidence | E3 | E0 + execution information |
| RQ3 Evidence | E4 | E0 + developer reference patch |

All prompts require the LLM to return only a JSON object:

```json
{"label": "correct|overfitting|uncertain", "confidence": 1, "explanation": "..."}
```

## Released Results

The `results/` directory contains released JSONL outputs for two models:

- `results/main_dataset_326(gpt-4o)/`
- `results/main_dataset_326(deepseek-v3)/`


## Rerunning LLM Judge Experiments

Inspect prompts without calling the API:

```bash
python scripts/run_llm_judge.py --setting S0 
```

Run one setting:

```bash
python scripts/run_llm_judge.py --setting S0 --model gpt-4o-2024-11-20
```

Run all metadata-leakage settings:

```bash
python scripts/run_llm_judge.py --group leakage --model gpt-4o-2024-11-20
```

Run all evidence settings:

```bash
python scripts/run_llm_judge.py --group evidence --model gpt-4o-2024-11-20
```

Run all settings:

```bash
python scripts/run_llm_judge.py --group all --model gpt-4o-2024-11-20
```

Resume an interrupted run:

```bash
python scripts/run_llm_judge.py --group all --resume --model gpt-4o-2024-11-20
```

By default, the runner writes `exp_*.jsonl` files under `results/`. Move regenerated files into a model-specific directory if you want to compare them with the released outputs.

## Cluster-Aware Robustness Analysis

Run:

```bash
python scripts/cluster_robustness.py
```

This script treats each bug/pair as the bootstrap cluster and compares core settings against S0. By default, it reads:

- `results/main_dataset_326(gpt-4o)/`
- `results/main_dataset_326(deepseek-v3)/`

It writes:

- `analysis/cluster_robustness/cluster_bootstrap_core.csv`
- `analysis/cluster_robustness/cluster_bootstrap_core.md`

You can also specify model result directories explicitly:

```bash
python scripts/cluster_robustness.py \
  --model-result "GPT-4o=results/main_dataset_326(gpt-4o)" \
  --model-result "DeepSeek-V3=results/main_dataset_326(deepseek-v3)"
```

## Reproducibility Notes

- The dataset is paired at the bug level: one correct and one overfitting patch per bug.
- S4R/S4F control metadata uses seed `20260512`.
- The default LLM temperature in `run_llm_judge.py` is `0`.




