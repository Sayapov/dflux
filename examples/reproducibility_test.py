#!/usr/bin/env python3
"""
Reproducibility Test — Are two identical telemetry runs exactly the same?

Runs GPT-2 twice with identical settings (greedy decoding, same prompt,
same device) and compares every single numerical field bit-for-bit.

If the forward pass is deterministic (which it should be with do_sample=False),
every field should match exactly — until we find where they don't.

Usage:
    python examples/reproducibility_test.py --model gpt2 --device cpu
    python examples/reproducibility_test.py --model gpt2 --device mps
    python examples/reproducibility_test.py --model EleutherAI/pythia-1.4b --device mps
"""

import argparse
import json
import gc
import sys
import os
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dflux.multiscale_telemetry import MultiScaleTelemetry, TelemetryConfig


def _detect_nested_keys(model_name: str) -> bool:
    """Peek at checkpoint key format without loading weights."""
    from huggingface_hub import hf_hub_download
    try:
        idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx_path) as f:
            first_key = next(iter(json.load(f)["weight_map"].keys()))
        return "language_model." in first_key
    except Exception:
        pass
    try:
        from safetensors import safe_open
        path = hf_hub_download(model_name, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            first_key = next(iter(f.keys()))
        return "language_model." in first_key
    except Exception:
        return True


def _load_qwen35(model_name: str, config, torch_dtype, device: str):
    """Qwen3.5 loading: promote config, detect key format, load/remap."""
    from transformers import Qwen3_5ForCausalLM

    # Skip deprecated config attrs during promotion
    _SKIP_ATTRS = {"use_return_dict", "output_hidden_states", "output_attentions",
                   "torchscript", "pruned_heads", "is_encoder_decoder"}
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        for attr in dir(text_cfg):
            if attr.startswith("_") or attr in _SKIP_ATTRS:
                continue
            try:
                object.__getattribute__(config, attr)
            except AttributeError:
                try:
                    val = getattr(text_cfg, attr)
                    if not callable(val):
                        setattr(config, attr, val)
                except Exception:
                    pass

    needs_remap = _detect_nested_keys(model_name)

    if not needs_remap:
        print(f"  Detected flat key format, using from_pretrained...")
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_name, config=config, dtype=torch_dtype,
            trust_remote_code=True,
        )
        return model.to(device)

    print(f"  Detected nested keys (language_model.*), using manual remap...")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as safe_load

    model = Qwen3_5ForCausalLM(config).to(dtype=torch_dtype)
    try:
        idx = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
        full_sd = {}
        for shard in set(weight_map.values()):
            path = hf_hub_download(model_name, shard)
            for k, v in safe_load(path, device="cpu").items():
                full_sd[k.replace("model.language_model.", "model.")] = v
    except Exception:
        path = hf_hub_download(model_name, "model.safetensors")
        full_sd = {k.replace("model.language_model.", "model."): v
                   for k, v in safe_load(path, device="cpu").items()}

    info = model.load_state_dict(full_sd, strict=False)
    loaded = len(full_sd) - len(info.unexpected_keys)
    print(f"  Remapped {loaded} weight tensors (language_model → flat)")
    if info.missing_keys:
        real_missing = [k for k in info.missing_keys if "lm_head" not in k]
        if real_missing:
            print(f"  Warning: {len(real_missing)} keys still missing")
    del full_sd
    gc.collect()
    return model.to(device)


def load_model(model_name: str, device: str, dtype: str = "float32"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"\n{'='*70}")
    print(f"Loading {model_name} on {device} ({dtype})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_5":
        model = _load_qwen35(model_name, config, torch_dtype, device)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch_dtype, trust_remote_code=True,
            ).to(device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, trust_remote_code=True,
            ).to(device)
    model.eval()

    return model, tokenizer


def run_single(model, tokenizer, device: str, prompt: str, max_new_tokens: int, run_id: int) -> Dict:
    """Run one telemetry capture and return the raw JSON-serializable data."""
    print(f"\n--- Run {run_id} ---")

    config = TelemetryConfig(
        logit_lens=True,
        logit_lens_top_k=5,
        cross_layer=True,
        mlp_internals=True,
        entropy_cascade=True,
        outlier_detection=True,
        capture_residuals=False,
        max_snapshots=512,
    )

    telem = MultiScaleTelemetry.from_model(model, tokenizer, cfg=config)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Ensure deterministic
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy — fully deterministic
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Generated: {generated_text[:100]}...")
    print(f"  Snapshots captured: {len(telem.snapshots)}")

    # Save to temp file and reload as dict for clean comparison
    tmp_path = f"/tmp/repro_run_{run_id}.json"
    telem.save(tmp_path)

    with open(tmp_path) as f:
        data = json.load(f)

    telem.detach()
    del telem

    return data, generated_text


def compare_values(v1: Any, v2: Any, path: str, diffs: List[Dict], tolerance: float = 0.0):
    """Recursively compare two values, recording all differences."""

    if type(v1) != type(v2):
        diffs.append({
            "path": path,
            "type": "type_mismatch",
            "run1": f"{type(v1).__name__}: {str(v1)[:50]}",
            "run2": f"{type(v2).__name__}: {str(v2)[:50]}",
        })
        return

    if isinstance(v1, dict):
        all_keys = set(list(v1.keys()) + list(v2.keys()))
        for k in sorted(all_keys):
            if k not in v1:
                diffs.append({"path": f"{path}.{k}", "type": "missing_in_run1"})
            elif k not in v2:
                diffs.append({"path": f"{path}.{k}", "type": "missing_in_run2"})
            else:
                compare_values(v1[k], v2[k], f"{path}.{k}", diffs, tolerance)

    elif isinstance(v1, list):
        if len(v1) != len(v2):
            diffs.append({
                "path": path,
                "type": "length_mismatch",
                "run1": len(v1),
                "run2": len(v2),
            })
            # Compare up to shorter length
            min_len = min(len(v1), len(v2))
        else:
            min_len = len(v1)

        for i in range(min_len):
            compare_values(v1[i], v2[i], f"{path}[{i}]", diffs, tolerance)

    elif isinstance(v1, (int, float)):
        if isinstance(v1, float) and (math.isnan(v1) or math.isnan(v2)):
            if not (math.isnan(v1) and math.isnan(v2)):
                diffs.append({
                    "path": path,
                    "type": "nan_mismatch",
                    "run1": v1,
                    "run2": v2,
                })
            return

        if v1 != v2:
            abs_diff = abs(v1 - v2)
            rel_diff = abs_diff / max(abs(v1), abs(v2), 1e-10)
            diffs.append({
                "path": path,
                "type": "value_mismatch",
                "run1": v1,
                "run2": v2,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            })

    elif isinstance(v1, str):
        if v1 != v2:
            diffs.append({
                "path": path,
                "type": "string_mismatch",
                "run1": v1[:50],
                "run2": v2[:50],
            })

    elif v1 is None and v2 is None:
        pass

    elif v1 != v2:
        diffs.append({
            "path": path,
            "type": "other_mismatch",
            "run1": str(v1)[:50],
            "run2": str(v2)[:50],
        })


def analyze_diffs(diffs: List[Dict]) -> Dict:
    """Categorize and summarize the differences found."""

    if not diffs:
        return {"total_diffs": 0, "verdict": "IDENTICAL"}

    # Separate by type
    value_mismatches = [d for d in diffs if d["type"] == "value_mismatch"]
    other_diffs = [d for d in diffs if d["type"] != "value_mismatch"]

    # For value mismatches, categorize by field and by magnitude
    field_diffs = {}
    for d in value_mismatches:
        # Extract the field name from the path
        # e.g., "snapshots[0].residual_norms[3]" -> "residual_norms"
        parts = d["path"].split(".")
        field = None
        for p in parts:
            if "[" in p:
                field = p.split("[")[0]
            else:
                field = p

        if field not in field_diffs:
            field_diffs[field] = {
                "count": 0,
                "max_abs_diff": 0,
                "max_rel_diff": 0,
                "examples": [],
            }

        field_diffs[field]["count"] += 1
        field_diffs[field]["max_abs_diff"] = max(field_diffs[field]["max_abs_diff"], d["abs_diff"])
        field_diffs[field]["max_rel_diff"] = max(field_diffs[field]["max_rel_diff"], d["rel_diff"])
        if len(field_diffs[field]["examples"]) < 3:
            field_diffs[field]["examples"].append(d)

    # Per-snapshot analysis — which tokens diverge?
    snapshot_diffs = {}
    for d in value_mismatches:
        # Extract snapshot index
        path = d["path"]
        if "snapshots[" in path:
            idx = int(path.split("snapshots[")[1].split("]")[0])
            if idx not in snapshot_diffs:
                snapshot_diffs[idx] = {"count": 0, "max_abs_diff": 0, "fields": set()}
            snapshot_diffs[idx]["count"] += 1
            snapshot_diffs[idx]["max_abs_diff"] = max(snapshot_diffs[idx]["max_abs_diff"], d["abs_diff"])
            # Extract field name
            after_snapshot = path.split(f"snapshots[{idx}].")[1] if f"snapshots[{idx}]." in path else "?"
            field_name = after_snapshot.split("[")[0]
            snapshot_diffs[idx]["fields"].add(field_name)

    # Per-layer analysis — which layers diverge?
    layer_diffs = {}
    for d in value_mismatches:
        path = d["path"]
        # Look for layer indices in things like residual_norms[5]
        # This is the index inside the per-layer arrays
        parts = path.split(".")
        for p in parts:
            if "[" in p and "]" in p:
                name = p.split("[")[0]
                idx = p.split("[")[1].split("]")[0]
                if name in ("residual_norms", "residual_deltas", "direction_changes",
                           "logit_lens_entropy", "logit_lens_top1_prob",
                           "mlp_norms", "attn_norms", "mlp_attn_ratio",
                           "mlp_dead_frac", "mlp_outlier_ratio",
                           "outlier_max_magnitude", "outlier_gini",
                           "entropy_reduction",
                           "dilution_survival", "dilution_energy_fraction",
                           "dilution_wasted_work", "dilution_cumulative_drift"):
                    layer = int(idx)
                    if layer not in layer_diffs:
                        layer_diffs[layer] = {"count": 0, "max_abs_diff": 0, "fields": set()}
                    layer_diffs[layer]["count"] += 1
                    layer_diffs[layer]["max_abs_diff"] = max(layer_diffs[layer]["max_abs_diff"], d["abs_diff"])
                    layer_diffs[layer]["fields"].add(name)

    return {
        "total_diffs": len(diffs),
        "value_mismatches": len(value_mismatches),
        "other_diffs": len(other_diffs),
        "other_diff_details": other_diffs[:10],
        "field_summary": field_diffs,
        "snapshot_summary": {k: {**v, "fields": list(v["fields"])} for k, v in sorted(snapshot_diffs.items())},
        "layer_summary": {k: {**v, "fields": list(v["fields"])} for k, v in sorted(layer_diffs.items())},
    }


def print_report(analysis: Dict, text1: str, text2: str):
    """Print a human-readable reproducibility report."""

    print(f"\n{'='*70}")
    print(f"  REPRODUCIBILITY REPORT")
    print(f"{'='*70}")

    # Text comparison
    if text1 == text2:
        print(f"\n  Generated text: IDENTICAL ✓")
    else:
        print(f"\n  Generated text: DIFFERENT ✗")
        # Find where they diverge
        for i, (c1, c2) in enumerate(zip(text1, text2)):
            if c1 != c2:
                print(f"  Diverges at character {i}:")
                print(f"    Run 1: ...{text1[max(0,i-20):i+20]}...")
                print(f"    Run 2: ...{text2[max(0,i-20):i+20]}...")
                break

    total = analysis["total_diffs"]
    print(f"\n  Total numerical differences: {total}")

    if total == 0:
        print(f"\n  ██████████████████████████████████████████████████")
        print(f"  ██  PERFECT MATCH — BIT-FOR-BIT IDENTICAL  ██")
        print(f"  ██████████████████████████████████████████████████")
        print(f"\n  Every single number in every single field across")
        print(f"  every single token's forward pass is identical.")
        print(f"  The telemetry is fully deterministic.")
        return

    print(f"  Value mismatches: {analysis['value_mismatches']}")
    if analysis["other_diffs"] > 0:
        print(f"  Structural diffs: {analysis['other_diffs']}")
        for d in analysis["other_diff_details"]:
            print(f"    {d['path']}: {d['type']}")

    # Field breakdown
    if analysis["field_summary"]:
        print(f"\n  ── Per-Field Breakdown ──")
        print(f"  {'Field':<30} {'Count':>6} {'Max Abs Diff':>14} {'Max Rel Diff':>14}")
        print(f"  {'─'*30} {'─'*6} {'─'*14} {'─'*14}")
        for field, info in sorted(analysis["field_summary"].items(),
                                   key=lambda x: x[1]["max_abs_diff"], reverse=True):
            print(f"  {field:<30} {info['count']:>6} {info['max_abs_diff']:>14.2e} {info['max_rel_diff']:>14.2e}")

    # Snapshot (token) breakdown
    if analysis["snapshot_summary"]:
        print(f"\n  ── Per-Token Breakdown ──")
        print(f"  {'Token':>6} {'Diffs':>6} {'Max Abs Diff':>14} {'Fields'}")
        print(f"  {'─'*6} {'─'*6} {'─'*14} {'─'*30}")
        for snap_idx, info in sorted(analysis["snapshot_summary"].items()):
            fields_str = ", ".join(sorted(info["fields"]))
            print(f"  {snap_idx:>6} {info['count']:>6} {info['max_abs_diff']:>14.2e} {fields_str}")

    # Layer breakdown
    if analysis["layer_summary"]:
        print(f"\n  ── Per-Layer Breakdown ──")
        print(f"  {'Layer':>6} {'Diffs':>6} {'Max Abs Diff':>14} {'Fields'}")
        print(f"  {'─'*6} {'─'*6} {'─'*14} {'─'*30}")
        for layer_idx, info in sorted(analysis["layer_summary"].items()):
            fields_str = ", ".join(sorted(info["fields"]))
            print(f"  {layer_idx:>6} {info['count']:>6} {info['max_abs_diff']:>14.2e} {fields_str}")

    # Verdict
    max_abs = max((d.get("abs_diff", 0) for d in analysis.get("field_summary", {}).values()), default=0)
    if isinstance(max_abs, dict):
        max_abs = max_abs.get("max_abs_diff", 0)
    else:
        max_abs = max((info["max_abs_diff"] for info in analysis["field_summary"].values()), default=0)

    print(f"\n  ── Verdict ──")
    if max_abs == 0:
        print(f"  IDENTICAL: No numerical differences detected")
    elif max_abs < 1e-7:
        print(f"  NEAR-IDENTICAL: Max diff {max_abs:.2e} (likely floating point noise)")
    elif max_abs < 1e-4:
        print(f"  MINOR DRIFT: Max diff {max_abs:.2e} (accumulation effects)")
    else:
        print(f"  SIGNIFICANT DRIFT: Max diff {max_abs:.2e}")
        print(f"  This suggests non-determinism in the computation path")


def count_numbers(obj, count=0):
    """Count total numerical values in a nested structure."""
    if isinstance(obj, (int, float)):
        return count + 1
    elif isinstance(obj, list):
        for item in obj:
            count = count_numbers(item, count)
        return count
    elif isinstance(obj, dict):
        for v in obj.values():
            count = count_numbers(v, count)
        return count
    return count


def compare_pair(data1, data2, text1, text2, run_a: int, run_b: int) -> Dict:
    """Compare two runs and return summary dict."""
    diffs = []
    compare_values(data1, data2, "root", diffs)

    non_timestamp_diffs = [d for d in diffs if "timestamp" not in d["path"]
                           and "capture_time" not in d["path"]
                           and "duration" not in d["path"]]

    analysis = analyze_diffs(non_timestamp_diffs)
    total_numbers = count_numbers(data1)
    matching = total_numbers - len(non_timestamp_diffs)

    return {
        "run_a": run_a,
        "run_b": run_b,
        "text_match": text1 == text2,
        "total_numbers": total_numbers,
        "total_diffs": len(non_timestamp_diffs),
        "matching": matching,
        "match_rate": matching / total_numbers if total_numbers > 0 else 1.0,
        "analysis": analysis,
        "diffs_sample": non_timestamp_diffs[:100],
    }


def print_multi_report(results: List[Dict], n_runs: int, model_name: str):
    """Print summary across all pairwise comparisons."""
    print(f"\n{'='*70}")
    print(f"  MULTI-RUN REPRODUCIBILITY REPORT")
    print(f"  Model: {model_name}")
    print(f"  Runs: {n_runs}")
    print(f"  Pairwise comparisons: {len(results)}")
    print(f"{'='*70}")

    all_identical = all(r["total_diffs"] == 0 for r in results)
    all_text_match = all(r["text_match"] for r in results)

    print(f"\n  Generated text across all runs: {'IDENTICAL' if all_text_match else 'DIFFERS'}")

    if all_identical:
        total_numbers = results[0]["total_numbers"] if results else 0
        print(f"\n  ██████████████████████████████████████████████████████████████")
        print(f"  ██  PERFECT MATCH — ALL {n_runs} RUNS BIT-FOR-BIT IDENTICAL  ██")
        print(f"  ██████████████████████████████████████████████████████████████")
        print(f"\n  {total_numbers:,} numerical values per run.")
        print(f"  {len(results)} pairwise comparisons.")
        print(f"  Zero differences in any pair.")
        print(f"\n  The telemetry is a deterministic fingerprint.")
        return

    # Pairwise matrix
    print(f"\n  ── Pairwise Diff Counts ──")
    print(f"  {'':>8}", end="")
    for i in range(1, n_runs + 1):
        print(f"  Run {i:>2}", end="")
    print()

    # Build lookup
    pair_lookup = {}
    for r in results:
        pair_lookup[(r["run_a"], r["run_b"])] = r["total_diffs"]

    for i in range(1, n_runs + 1):
        print(f"  Run {i:>2}", end="")
        for j in range(1, n_runs + 1):
            if i == j:
                print(f"  {'—':>6}", end="")
            else:
                key = (min(i, j), max(i, j))
                d = pair_lookup.get(key, "?")
                if d == 0:
                    print(f"  {'✓':>6}", end="")
                else:
                    print(f"  {d:>6}", end="")
        print()

    # Which pairs differ?
    differing = [r for r in results if r["total_diffs"] > 0]
    if differing:
        print(f"\n  ── Differing Pairs ──")
        for r in differing:
            print(f"\n  Run {r['run_a']} vs Run {r['run_b']}: {r['total_diffs']} diffs")
            analysis = r["analysis"]
            if analysis.get("field_summary"):
                print(f"  {'Field':<30} {'Count':>6} {'Max Abs':>14} {'Max Rel':>14}")
                print(f"  {'─'*30} {'─'*6} {'─'*14} {'─'*14}")
                for field, info in sorted(analysis["field_summary"].items(),
                                          key=lambda x: x[1]["max_abs_diff"], reverse=True):
                    print(f"  {field:<30} {info['count']:>6} {info['max_abs_diff']:>14.2e} {info['max_rel_diff']:>14.2e}")

            if analysis.get("layer_summary"):
                print(f"\n  Per-layer:")
                for layer_idx, info in sorted(analysis["layer_summary"].items()):
                    fields_str = ", ".join(sorted(info["fields"]))
                    print(f"    L{layer_idx:>2}: {info['count']:>4} diffs, max {info['max_abs_diff']:.2e}  [{fields_str}]")

    # Overall verdict
    max_diff = max((r["total_diffs"] for r in results), default=0)
    max_abs = 0
    for r in results:
        for field_info in r["analysis"].get("field_summary", {}).values():
            max_abs = max(max_abs, field_info["max_abs_diff"])

    print(f"\n  ── Verdict ──")
    if max_abs == 0:
        print(f"  ALL IDENTICAL across {n_runs} runs")
    elif max_abs < 1e-7:
        print(f"  NEAR-IDENTICAL: Max diff {max_abs:.2e} across all pairs (float noise)")
    elif max_abs < 1e-4:
        print(f"  MINOR DRIFT: Max diff {max_abs:.2e} — accumulation across {n_runs} runs")
    else:
        print(f"  SIGNIFICANT DRIFT: Max diff {max_abs:.2e}")
        print(f"  Non-determinism detected in the computation path")


def main():
    parser = argparse.ArgumentParser(description="Telemetry Reproducibility Test")
    parser.add_argument("--model", default="gpt2", help="Model to test")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--prompt", default="The meaning of life is", help="Input prompt")
    parser.add_argument("--tokens", type=int, default=32, help="Tokens to generate")
    parser.add_argument("--runs", type=int, default=2, help="Number of runs (compared pairwise)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (bfloat16 for large models)")
    parser.add_argument("--output", default=None, help="Save detailed diff JSON")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device, args.dtype)

    # Run N times
    runs_data = []
    runs_text = []
    for i in range(1, args.runs + 1):
        print(f"\n{'='*70}")
        print(f"  RUN {i} of {args.runs}")
        print(f"{'='*70}")
        data, text = run_single(model, tokenizer, args.device, args.prompt, args.tokens, i)
        runs_data.append(data)
        runs_text.append(text)

    # Pairwise comparison
    print(f"\n{'='*70}")
    print(f"  COMPARING ALL PAIRS...")
    print(f"{'='*70}")

    results = []
    for i in range(len(runs_data)):
        for j in range(i + 1, len(runs_data)):
            print(f"  Comparing run {i+1} vs run {j+1}...", end=" ")
            result = compare_pair(runs_data[i], runs_data[j],
                                  runs_text[i], runs_text[j], i + 1, j + 1)
            results.append(result)
            status = "IDENTICAL" if result["total_diffs"] == 0 else f"{result['total_diffs']} diffs"
            print(status)

    print_multi_report(results, args.runs, args.model)

    # Save detailed output
    safe_name = args.model.replace("/", "-")
    repro_dir = Path("data/reproducibility")
    repro_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(repro_dir / f"reproducibility_{safe_name}.json")

    report = {
        "model": args.model,
        "device": args.device,
        "prompt": args.prompt,
        "max_new_tokens": args.tokens,
        "n_runs": args.runs,
        "all_identical": all(r["total_diffs"] == 0 for r in results),
        "pairwise_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved to: {output_path}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    elif args.device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
