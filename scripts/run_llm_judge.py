"""
LLM Judge runner for ESEM APR patch correctness study.

Usage:
  # Run a single setting on all patches (main experiment)
  python scripts/run_llm_judge.py --setting S0

  # Run multiple settings
  python scripts/run_llm_judge.py --setting S0 S1 S2 S3 S4 S5

  # Run all leakage settings
  python scripts/run_llm_judge.py --group leakage

  # Run all evidence settings
  python scripts/run_llm_judge.py --group evidence

  # Run all settings
  python scripts/run_llm_judge.py --group all

  # Pilot mode: first 30 patches, with consistency check (3 repeats)
  python scripts/run_llm_judge.py --setting S0 --pilot

  # Dry run: print prompts without calling API
  python scripts/run_llm_judge.py --setting S0 --dry-run --limit 2

  # Resume from where a previous run stopped
  python scripts/run_llm_judge.py --setting S0 --resume

  # Use a specific model
  python scripts/run_llm_judge.py --setting S0 --model gpt-4o-2024-11-20

Requires:
  pip install openai
  export OPENAI_API_KEY=sk-...
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from prompt_builder import (
    SYSTEM_PROMPT,
    SETTING_BUILDERS,
    ALL_SETTINGS,
    LEAKAGE_SETTINGS,
    EVIDENCE_SETTINGS,
    build_prompt,
)

DATA_FILE = PROJECT_DIR / "data" / "main_dataset_326.jsonl"
RESULTS_DIR = PROJECT_DIR / "results"
PILOT_DIR = RESULTS_DIR / "pilot"

DEFAULT_MODEL = "gpt-4o-2024-11-20"
# DEFAULT_MODEL = "DeepSeek-V3"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 512
MAX_RETRIES = 2
RETRY_DELAY = 5
RATE_LIMIT_DELAY = 1.0

PILOT_COUNT = 30
CONTROL_METADATA_SEED = 20260512
FAKE_BUG_BASE = 9000


def load_dataset(path, limit=None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def attach_control_metadata(records, seed=CONTROL_METADATA_SEED):
    """Attach deterministic shuffled and fake metadata used by S4R/S4F."""
    enriched = [copy.deepcopy(r) for r in records]
    source = enriched[:]
    shuffled = enriched[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if len(shuffled) > 1:
        fixed_points = [
            i for i, (record, donor) in enumerate(zip(source, shuffled))
            if record.get("patch_id") == donor.get("patch_id")
        ]
        for i in fixed_points:
            j = (i + 1) % len(shuffled)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    for i, (record, donor) in enumerate(zip(enriched, shuffled)):
        record["shuffled_meta_bug_id"] = donor.get("meta_bug_id", "")
        record["shuffled_meta_project"] = donor.get("meta_project", "")
        record["shuffled_meta_tool"] = donor.get("meta_tool", "")
        record["fake_meta_bug_id"] = f"Fake-{FAKE_BUG_BASE + i}"
    return enriched


def select_pilot_records(records, count=PILOT_COUNT):
    """Select first count/2 correct + first count/2 overfitting patches."""
    correct = [r for r in records if r["ground_truth"] == "correct"]
    overfit = [r for r in records if r["ground_truth"] == "overfitting"]
    half = count // 2
    return correct[:half] + overfit[:half]


def parse_llm_response(text):
    """Parse JSON from LLM response, handling common formatting issues."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        result = json.loads(text)
        return result, None
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"label"\s*:.*?\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return result, None
        except json.JSONDecodeError:
            pass

    return None, f"Failed to parse JSON from response: {text[:200]}"


def validate_result(result):
    """Validate parsed result has required fields."""
    if not isinstance(result, dict):
        return None, "Response is not a JSON object"
    label = result.get("label", "")
    if label not in ("correct", "overfitting", "uncertain"):
        return None, f"Invalid label: {label}"
    conf = result.get("confidence")
    if conf is not None:
        try:
            conf = int(conf)
            if not 1 <= conf <= 5:
                result["confidence"] = max(1, min(5, conf))
        except (ValueError, TypeError):
            result["confidence"] = 3
    else:
        result["confidence"] = 3
    if "explanation" not in result:
        result["explanation"] = ""
    return result, None


def call_openai(prompt, system_prompt, model, temperature, max_tokens):
    """Call OpenAI API. Returns (response_text, error_string)."""
    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed. Run: pip install openai", {}

    try:
        from config import OPENAI_API_KEY, OPENAI_BASE_URL
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    except ImportError:
        client = OpenAI()
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return text, None, usage
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < MAX_RETRIES - 1:
                print(f"    API error (attempt {attempt+1}): {err_str[:100]}")
                time.sleep(RETRY_DELAY)
                continue
            return None, f"API error after {MAX_RETRIES} attempts: {err_str[:200]}", {}


def load_completed(output_path):
    """Load already-completed patch IDs from an output file for resume."""
    completed = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add(rec.get("patch_id", ""))
                except json.JSONDecodeError:
                    continue
    return completed


def run_experiment(records, setting, model, temperature, max_tokens,
                   output_path, dry_run=False, resume=False,
                   repeat=1):
    """Run LLM Judge on all records for a given setting."""
    completed = load_completed(output_path) if resume else set()
    if completed:
        print(f"  Resuming: {len(completed)} already completed")

    mode = "a" if resume else "w"
    total = len(records)
    success = 0
    parse_fail = 0
    total_tokens = 0

    with open(output_path, mode, encoding="utf-8") as out_f:
        for i, record in enumerate(records):
            patch_id = record["patch_id"]
            if patch_id in completed:
                continue

            prompt = build_prompt(record, setting)

            if dry_run:
                print(f"\n{'='*60}")
                print(f"[{i+1}/{total}] {patch_id} | setting={setting}")
                print(f"{'='*60}")
                print(f"SYSTEM: {SYSTEM_PROMPT[:80]}...")
                print(f"\nUSER PROMPT ({len(prompt)} chars):")
                print(prompt[:500])
                if len(prompt) > 500:
                    print(f"... ({len(prompt)-500} more chars)")
                continue

            for rep in range(repeat):
                rep_id = f"rep{rep}" if repeat > 1 else None
                print(f"  [{i+1}/{total}] {patch_id} setting={setting}"
                      + (f" {rep_id}" if rep_id else ""), end="")

                text, error, usage = call_openai(
                    prompt, SYSTEM_PROMPT, model, temperature, max_tokens
                )

                if error:
                    result_record = {
                        "patch_id": patch_id,
                        "pair_id": record["pair_id"],
                        "setting": setting,
                        "ground_truth": record["ground_truth"],
                        "error": error,
                    }
                    if rep_id:
                        result_record["repeat"] = rep
                    out_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    parse_fail += 1
                    print(f" ERROR: {error[:60]}")
                    continue

                total_tokens += usage.get("total_tokens", 0)
                parsed, parse_err = parse_llm_response(text)

                if parse_err:
                    # Retry once with a nudge
                    nudge = (
                        "Your previous response was not valid JSON. "
                        "Please respond with ONLY a JSON object: "
                        '{"label": "correct|overfitting|uncertain", '
                        '"confidence": 1-5, "explanation": "..."}'
                    )
                    text2, error2, usage2 = call_openai(
                        prompt + "\n\n" + nudge, SYSTEM_PROMPT,
                        model, temperature, max_tokens
                    )
                    if not error2:
                        total_tokens += usage2.get("total_tokens", 0)
                        parsed, parse_err = parse_llm_response(text2)

                if parsed:
                    parsed, val_err = validate_result(parsed)

                if parsed:
                    result_record = {
                        "patch_id": patch_id,
                        "pair_id": record["pair_id"],
                        "setting": setting,
                        "ground_truth": record["ground_truth"],
                        "llm_label": parsed["label"],
                        "llm_confidence": parsed["confidence"],
                        "llm_explanation": parsed.get("explanation", ""),
                        "meta_project": record.get("meta_project", ""),
                        "meta_tool": record.get("meta_tool", ""),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    }
                    if rep_id:
                        result_record["repeat"] = rep
                    out_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    success += 1
                    label = parsed["label"]
                    conf = parsed["confidence"]
                    gt = record["ground_truth"]
                    match = "OK" if label == gt else "MISS"
                    print(f" -> {label} (conf={conf}) [{match}]")
                else:
                    result_record = {
                        "patch_id": patch_id,
                        "pair_id": record["pair_id"],
                        "setting": setting,
                        "ground_truth": record["ground_truth"],
                        "error": parse_err or val_err or "unknown parse error",
                        "raw_response": text[:500] if text else "",
                    }
                    if rep_id:
                        result_record["repeat"] = rep
                    out_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    parse_fail += 1
                    print(f" PARSE_FAIL")

                time.sleep(RATE_LIMIT_DELAY)

    return success, parse_fail, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Judge for APR patch correctness assessment"
    )
    parser.add_argument(
        "--setting", nargs="+",
        help="Prompt settings to run (e.g., S0 S1 E2)"
    )
    parser.add_argument(
        "--group", choices=["leakage", "evidence", "all"],
        help="Run a predefined group of settings"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run pilot study (30 patches, 3 repeats for S0)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run")
    parser.add_argument("--limit", type=int, help="Limit number of records")
    parser.add_argument("--data", type=str, default=str(DATA_FILE),
                        help="Path to dataset JSONL")
    args = parser.parse_args()

    # Determine settings to run
    if args.group:
        if args.group == "leakage":
            settings = LEAKAGE_SETTINGS
        elif args.group == "evidence":
            settings = EVIDENCE_SETTINGS
        else:
            settings = ALL_SETTINGS
    elif args.setting:
        settings = args.setting
    else:
        settings = ["S0"]

    for s in settings:
        if s not in SETTING_BUILDERS:
            print(f"ERROR: Unknown setting '{s}'. Available: {ALL_SETTINGS}")
            sys.exit(1)

    # Load data
    data_path = Path(args.data)
    records = load_dataset(data_path, limit=args.limit)
    records = attach_control_metadata(records)

    if args.pilot:
        records = select_pilot_records(records, PILOT_COUNT)
        print(f"PILOT MODE: {len(records)} patches selected "
              f"({sum(1 for r in records if r['ground_truth']=='correct')} correct, "
              f"{sum(1 for r in records if r['ground_truth']=='overfitting')} overfitting)")

    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Dataset: {data_path} ({len(records)} records)")
    print(f"Settings: {settings}")
    print(f"Dry run: {args.dry_run}")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PILOT_DIR, exist_ok=True)

    grand_success = 0
    grand_fail = 0
    grand_tokens = 0

    for setting in settings:
        print(f"\n{'='*60}")
        print(f"Running setting: {setting}")
        print(f"{'='*60}")

        if args.pilot:
            out_dir = PILOT_DIR
            repeat = 3 if setting == "S0" else 1
        else:
            out_dir = RESULTS_DIR
            repeat = 1

        output_path = out_dir / f"exp_{setting.lower()}.jsonl"

        success, fail, tokens = run_experiment(
            records=records,
            setting=setting,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_path=output_path,
            dry_run=args.dry_run,
            resume=args.resume,
            repeat=repeat,
        )

        grand_success += success
        grand_fail += fail
        grand_tokens += tokens

        print(f"\n  Setting {setting} complete:")
        print(f"    Success: {success}")
        print(f"    Parse failures: {fail}")
        print(f"    Tokens used: {grand_tokens:,}")
        print(f"    Output: {output_path}")

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"ALL DONE")
        print(f"  Total success: {grand_success}")
        print(f"  Total parse failures: {grand_fail}")
        print(f"  Total tokens: {grand_tokens:,}")
        est_cost = grand_tokens / 1_000_000 * 5
        print(f"  Estimated cost: ~${est_cost:.2f}")


if __name__ == "__main__":
    main()
