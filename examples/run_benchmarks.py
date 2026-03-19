#!/usr/bin/env python3
"""Run lm-eval benchmarks with and without D-Flux governor profiles.

Usage:
    # Baseline vs governed
    python examples/run_benchmarks.py \
        --model Qwen/Qwen3.5-9B-Base \
        --profile profiles/qwen35_9b_reasoning_dilution_survival_0.3.json \
        --tasks hellaswag,arc_challenge \
        --device mps --dtype bfloat16

    # Governed only
    python examples/run_benchmarks.py \
        --model Qwen/Qwen3.5-9B-Base \
        --profile profiles/qwen35_9b_reasoning_dilution_survival_0.3.json \
        --tasks hellaswag \
        --skip-baseline --device mps --dtype bfloat16

    # Baseline only (no profile)
    python examples/run_benchmarks.py \
        --model Qwen/Qwen3.5-9B-Base \
        --tasks hellaswag \
        --device mps --dtype bfloat16
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Ensure dflux is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lm_eval
import dflux.eval_model  # registers dflux-governed model


def run_eval(model_name, tasks, profile_path=None, device="mps",
             dtype="bfloat16", batch_size=1, num_fewshot=0, limit=None):
    """Run lm-eval and return results dict."""

    if profile_path:
        model_type = "dflux-governed"
        model_args = {
            "pretrained": model_name,
            "profile_path": profile_path,
            "trust_remote_code": True,
            "dtype": dtype,
        }
    else:
        # Use dflux-governed with no profile (handles Qwen3.5 loading)
        model_type = "dflux-governed"
        model_args = {
            "pretrained": model_name,
            "trust_remote_code": True,
            "dtype": dtype,
        }

    results = lm_eval.simple_evaluate(
        model=model_type,
        model_args=model_args,
        tasks=tasks.split(",") if isinstance(tasks, str) else tasks,
        batch_size=batch_size,
        device=device,
        num_fewshot=num_fewshot,
        limit=limit,
    )
    return results


def print_comparison(baseline_results, governed_results, tasks):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARISON: Baseline vs Governed")
    print("=" * 70)

    task_list = tasks.split(",") if isinstance(tasks, str) else tasks

    for task in task_list:
        b = baseline_results["results"].get(task, {})
        g = governed_results["results"].get(task, {})

        print(f"\n  Task: {task}")
        print(f"  {'Metric':<30} {'Baseline':>10} {'Governed':>10} {'Delta':>10}")
        print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}")

        # Find common metrics
        all_keys = set(list(b.keys()) + list(g.keys()))
        for key in sorted(all_keys):
            if key.startswith("alias") or key == "group_name":
                continue
            bv = b.get(key)
            gv = g.get(key)
            if bv is None or gv is None:
                continue
            if not isinstance(bv, (int, float)) or not isinstance(gv, (int, float)):
                continue

            delta = gv - bv
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {key:<30} {bv:>10.4f} {gv:>10.4f} {delta:>+9.4f} {arrow}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Run D-Flux governed benchmarks")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--profile", default=None, help="Profile JSON path")
    parser.add_argument("--tasks", default="hellaswag", help="Comma-separated task names")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (testing only)")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--output-dir", default="data/benchmarks")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]

    # Run baseline
    baseline_results = None
    if not args.skip_baseline:
        print("\n" + "=" * 70)
        print("  PHASE 1: BASELINE (no profile)")
        print("=" * 70)
        baseline_results = run_eval(
            args.model, args.tasks, profile_path=None,
            device=args.device, dtype=args.dtype,
            batch_size=args.batch_size, num_fewshot=args.num_fewshot,
            limit=args.limit,
        )
        out = os.path.join(args.output_dir, f"{model_short}_baseline_{timestamp}.json")
        with open(out, "w") as f:
            json.dump(baseline_results["results"], f, indent=2, default=str)
        print(f"\n  Saved: {out}")

        # Print baseline results
        for task, metrics in baseline_results["results"].items():
            print(f"\n  {task}:")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")

    # Run governed
    governed_results = None
    if args.profile:
        print("\n" + "=" * 70)
        print(f"  PHASE 2: GOVERNED ({os.path.basename(args.profile)})")
        print("=" * 70)
        governed_results = run_eval(
            args.model, args.tasks, profile_path=args.profile,
            device=args.device, dtype=args.dtype,
            batch_size=args.batch_size, num_fewshot=args.num_fewshot,
            limit=args.limit,
        )
        profile_short = os.path.splitext(os.path.basename(args.profile))[0]
        out = os.path.join(args.output_dir, f"{model_short}_governed_{profile_short}_{timestamp}.json")
        with open(out, "w") as f:
            json.dump(governed_results["results"], f, indent=2, default=str)
        print(f"\n  Saved: {out}")

    # Comparison
    if baseline_results and governed_results:
        print_comparison(baseline_results, governed_results, args.tasks)

    # Print final summary
    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
