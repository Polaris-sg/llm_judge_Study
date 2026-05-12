"""
Cluster-aware robustness analysis for core prompt comparisons.

The unit of resampling is the bug/pair cluster. Each Defects4J bug
contributes one correct and one overfitting patch, so cluster bootstrap
keeps those paired observations together when estimating uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "analysis" / "cluster_robustness"
CORE_SETTINGS = ["S4", "S5", "E2", "E4"]
DEFAULT_MODEL_RESULT_DIRS = {
    "GPT-4o": RESULTS_DIR / "main_dataset_326(gpt-4o)",
    "DeepSeek-V3": RESULTS_DIR / "main_dataset_326(deepseek-v3)",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" in rec:
                raise ValueError(f"{path} contains API error for {rec.get('patch_id')}")
            records[rec["patch_id"]] = rec
    return records


def setting_path(results_dir: Path, setting: str) -> Path:
    if setting == "E0":
        setting = "S0"
    path = results_dir / f"exp_{setting.lower()}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_setting(results_dir: Path, setting: str) -> dict[str, dict]:
    return load_jsonl(setting_path(results_dir, setting))


def is_correct(rec: dict) -> bool:
    return rec["llm_label"] == rec["ground_truth"]


def metrics_for_pairs(base: list[dict], target: list[dict]) -> dict[str, float]:
    n = len(base)
    if n != len(target):
        raise ValueError("base/target length mismatch")
    base_acc = sum(is_correct(r) for r in base) / n
    target_acc = sum(is_correct(r) for r in target) / n

    base_over = [r for r in base if r["ground_truth"] == "overfitting"]
    target_over = [r for r in target if r["ground_truth"] == "overfitting"]
    base_fpr = sum(r["llm_label"] == "correct" for r in base_over) / len(base_over)
    target_fpr = sum(r["llm_label"] == "correct" for r in target_over) / len(target_over)

    better = 0
    worse = 0
    for b, t in zip(base, target):
        b_ok = is_correct(b)
        t_ok = is_correct(t)
        if (not b_ok) and t_ok:
            better += 1
        elif b_ok and (not t_ok):
            worse += 1

    return {
        "accuracy_delta": target_acc - base_acc,
        "fpr_delta": target_fpr - base_fpr,
        "better_minus_worse_rate": (better - worse) / n,
        "better_n": better,
        "worse_n": worse,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cluster_bootstrap(base_records: dict[str, dict], target_records: dict[str, dict],
                      iterations: int, seed: int) -> dict[str, dict[str, float]]:
    ids = sorted(base_records)
    if set(ids) != set(target_records):
        raise ValueError("Patch set mismatch")

    clusters: dict[str, list[str]] = defaultdict(list)
    for patch_id in ids:
        clusters[str(base_records[patch_id]["pair_id"])].append(patch_id)
    cluster_ids = sorted(clusters)

    observed = metrics_for_pairs(
        [base_records[pid] for pid in ids],
        [target_records[pid] for pid in ids],
    )

    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        sampled_patch_ids = []
        for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
            sampled_patch_ids.extend(clusters[cluster_id])
        m = metrics_for_pairs(
            [base_records[pid] for pid in sampled_patch_ids],
            [target_records[pid] for pid in sampled_patch_ids],
        )
        for key in ("accuracy_delta", "fpr_delta", "better_minus_worse_rate"):
            samples[key].append(m[key])

    summary = {}
    for key, values in samples.items():
        summary[key] = {
            "observed": observed[key],
            "ci_low": percentile(values, 0.025),
            "ci_high": percentile(values, 0.975),
        }
    summary["counts"] = {
        "better_n": observed["better_n"],
        "worse_n": observed["worse_n"],
        "clusters": len(cluster_ids),
        "patches": len(ids),
    }
    return summary


def analyze_model(model_name: str, results_dir: Path, settings: list[str],
                  iterations: int, seed: int) -> list[dict]:
    base = load_setting(results_dir, "S0")
    rows = []
    for offset, setting in enumerate(settings):
        target = load_setting(results_dir, setting)
        summary = cluster_bootstrap(base, target, iterations, seed + offset)
        counts = summary["counts"]
        row = {
            "model": model_name,
            "setting": setting,
            "clusters": counts["clusters"],
            "patches": counts["patches"],
            "better_n": counts["better_n"],
            "worse_n": counts["worse_n"],
        }
        for metric in ("accuracy_delta", "fpr_delta", "better_minus_worse_rate"):
            row[f"{metric}_observed"] = summary[metric]["observed"]
            row[f"{metric}_ci_low"] = summary[metric]["ci_low"]
            row[f"{metric}_ci_high"] = summary[metric]["ci_high"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.1f}"


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Cluster-Aware Robustness Analysis",
        "",
        "Cluster bootstrap resamples 163 bug/pair clusters with replacement.",
        "Intervals are percentile 95% confidence intervals over bootstrap samples.",
        "",
        "| Model | Setting | Better/Worse | Delta Acc. (95% CI, pp) | Delta FPR (95% CI, pp) | Better-Worse Rate (95% CI, pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {setting} | {better_n}/{worse_n} | "
            "{acc} [{acc_l}, {acc_h}] | "
            "{fpr} [{fpr_l}, {fpr_h}] | "
            "{bw} [{bw_l}, {bw_h}] |".format(
                model=row["model"],
                setting=row["setting"],
                better_n=row["better_n"],
                worse_n=row["worse_n"],
                acc=fmt_pct(row["accuracy_delta_observed"]),
                acc_l=fmt_pct(row["accuracy_delta_ci_low"]),
                acc_h=fmt_pct(row["accuracy_delta_ci_high"]),
                fpr=fmt_pct(row["fpr_delta_observed"]),
                fpr_l=fmt_pct(row["fpr_delta_ci_low"]),
                fpr_h=fmt_pct(row["fpr_delta_ci_high"]),
                bw=fmt_pct(row["better_minus_worse_rate_observed"]),
                bw_l=fmt_pct(row["better_minus_worse_rate_ci_low"]),
                bw_h=fmt_pct(row["better_minus_worse_rate_ci_high"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--setting", nargs="+", default=CORE_SETTINGS)
    parser.add_argument(
        "--model-result",
        action="append",
        default=[],
        help="Model result spec NAME=PATH. Defaults to GPT-4o and DeepSeek-V3.",
    )
    args = parser.parse_args()

    if args.model_result:
        model_dirs = {}
        for spec in args.model_result:
            name, path = spec.split("=", 1)
            model_dirs[name] = Path(path)
    else:
        model_dirs = DEFAULT_MODEL_RESULT_DIRS

    rows = []
    for i, (model_name, results_dir) in enumerate(model_dirs.items()):
        rows.extend(
            analyze_model(
                model_name,
                results_dir,
                args.setting,
                args.iterations,
                args.seed + 1000 * i,
            )
        )

    write_csv(args.out_dir / "cluster_bootstrap_core.csv", rows)
    write_markdown(args.out_dir / "cluster_bootstrap_core.md", rows)
    print(f"Wrote {args.out_dir / 'cluster_bootstrap_core.csv'}")
    print(f"Wrote {args.out_dir / 'cluster_bootstrap_core.md'}")


if __name__ == "__main__":
    main()
