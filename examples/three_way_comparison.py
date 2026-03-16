#!/usr/bin/env python3
"""
Three-way telemetry comparison: base → instruct → reasoning-distilled.

Runs identical prompts through N models and produces:
  1. Per-signal numerical diff between each consecutive pair
  2. Per-layer breakdown with layer type annotations
  3. Combined visualization with all models overlaid
  4. Pairwise diff reports

Designed for the Qwen3.5-9B family:
  base       → Qwen/Qwen3.5-9B-Base
  instruct   → Qwen/Qwen3.5-9B
  reasoning  → Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled

Usage:
    # Full three-way on MPS (each model loads → runs → frees memory)
    python examples/three_way_comparison.py \
        --models Qwen/Qwen3.5-9B-Base \
                 Qwen/Qwen3.5-9B \
                 Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled \
        --labels base instruct reasoning \
        --device mps --dtype bfloat16

    # Two-way (backward compatible)
    python examples/three_way_comparison.py \
        --models Qwen/Qwen3.5-9B-Base Qwen/Qwen3.5-9B \
        --labels base instruct \
        --device mps --dtype bfloat16

    # From existing telemetry JSONs
    python examples/three_way_comparison.py \
        --from-json data/telemetry/base.json data/telemetry/instruct.json data/telemetry/reasoning.json \
        --labels base instruct reasoning
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dflux.multiscale_telemetry import MultiScaleTelemetry, TelemetryConfig


# ── Model loading ─────────────────────────────────────────────────

def load_model(model_name: str, device: str, dtype: str = "float32"):
    """Load model and tokenizer, handling Qwen3.5 weight remapping."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"\n  Loading {model_name} ({dtype})...")
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
        except AttributeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch_dtype, trust_remote_code=True,
            ).to(device)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {n_params:.0f}M params on {device} ({torch_dtype})")
    return model, tokenizer


def _detect_nested_keys(model_name: str) -> bool:
    """Peek at checkpoint key format without loading weights.

    Returns True if keys contain 'language_model.' (official Qwen3.5 format),
    False if keys are flat (fine-tunes from Unsloth/PEFT).
    """
    from huggingface_hub import hf_hub_download

    try:
        idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx_path) as f:
            first_key = next(iter(json.load(f)["weight_map"].keys()))
        return "language_model." in first_key
    except Exception:
        pass

    # Single-file checkpoint: peek at key names without loading tensors
    try:
        from safetensors import safe_open
        path = hf_hub_download(model_name, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            first_key = f.keys()[0] if hasattr(f.keys(), '__getitem__') else next(iter(f.keys()))
        return "language_model." in first_key
    except Exception:
        pass

    # Can't determine — assume nested (the safer default)
    return True


def _load_qwen35(model_name: str, config, torch_dtype, device: str):
    """Qwen3.5 loading with config promotion + weight key remapping.

    Official Qwen3.5 checkpoints nest weights under model.language_model.*
    but the promoted config builds model.layers.* (flat).  Fine-tunes from
    Unsloth/PEFT typically save in the flat format already.

    We peek at checkpoint key format first to choose the right loading path,
    avoiding the noisy LOAD REPORT from a mismatched from_pretrained attempt.
    """
    from transformers import Qwen3_5ForCausalLM

    # Promote text_config fields to top-level (skip deprecated attrs)
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

    # Peek at checkpoint key format
    needs_remap = _detect_nested_keys(model_name)

    if not needs_remap:
        # Flat key format (Unsloth/PEFT fine-tunes) — from_pretrained works directly
        print(f"  Detected flat key format, using from_pretrained...")
        try:
            model = Qwen3_5ForCausalLM.from_pretrained(
                model_name, config=config, dtype=torch_dtype,
                trust_remote_code=True,
            )
            print(f"  Loaded via from_pretrained (flat key format)")
            return model.to(device)
        except Exception as e:
            print(f"  from_pretrained failed ({e}), falling back to manual remap...")

    # Nested key format (official checkpoints) — manual load + remap
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


# ── Telemetry capture ─────────────────────────────────────────────

def run_telemetry(model, tokenizer, prompt: str, max_tokens: int,
                  device: str, label: str) -> Tuple[dict, str]:
    """Run telemetry capture and return (json_data, generated_text)."""
    print(f"\n  Running telemetry: {label}")

    cfg = TelemetryConfig(
        logit_lens=True,
        logit_lens_top_k=5,
        cross_layer=True,
        mlp_internals=True,
        entropy_cascade=True,
        outlier_detection=True,
    )

    telem = MultiScaleTelemetry.from_model(model, tokenizer, cfg=cfg)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch.manual_seed(42)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Generated: {text[:120]}...")
    print(f"  Snapshots: {len(telem.snapshots)}")

    tmp = f"/tmp/nway_{label.replace(' ', '_')}.json"
    telem.save(tmp)
    with open(tmp) as f:
        data = json.load(f)

    telem.detach()
    return data, text


def free_model(model, tokenizer, device: str):
    """Free model memory."""
    del model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


# ── Comparison ────────────────────────────────────────────────────

SIGNALS = [
    "residual_norms", "residual_deltas", "direction_changes",
    "logit_lens_entropy", "logit_lens_top1_prob",
    "mlp_norms", "attn_norms", "mlp_attn_ratio",
    "mlp_dead_frac", "mlp_outlier_ratio",
    "entropy_reduction",
    "outlier_max_magnitude", "outlier_gini",
    "dilution_survival", "dilution_energy_frac",
    "dilution_wasted_work", "dilution_cumulative_drift",
]


def extract_signal(data: dict, signal: str) -> Optional[np.ndarray]:
    """Extract a per-layer mean signal from telemetry data."""
    agg = data.get("aggregate", {})
    vals = agg.get(f"{signal}_mean")
    if vals is None:
        return None
    return np.array(vals)


def pairwise_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute diff stats between two per-layer arrays."""
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    abs_diff = np.abs(b - a)
    rel_diff = abs_diff / (np.abs(a) + 1e-10)
    return {
        "abs_diff": abs_diff.tolist(),
        "rel_diff": rel_diff.tolist(),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "max_abs_layer": int(np.argmax(abs_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
    }


# ── Reporting ─────────────────────────────────────────────────────

def print_nway_report(all_data: List[dict], labels: List[str],
                      texts: List[str]):
    """Print comparison report for N models."""
    n = len(all_data)
    n_layers = all_data[0]["n_layers"]
    layer_types = all_data[0].get("layer_types")

    print(f"\n{'='*80}")
    print(f"  {n}-WAY TELEMETRY COMPARISON")
    print(f"{'='*80}")
    for i, (data, label) in enumerate(zip(all_data, labels)):
        model_name = data.get("model", label)
        print(f"  [{label}] {model_name} ({data['n_layers']} layers)")
    if layer_types:
        n_full = sum(1 for lt in layer_types if lt == "full_attention")
        n_lin = sum(1 for lt in layer_types if lt == "linear_attention")
        print(f"  Architecture: hybrid ({n_full} full attention, {n_lin} linear attention)")

    # Generated text
    print(f"\n  ── Generated Text ──")
    for label, text in zip(labels, texts):
        print(f"  [{label}] {text[:100]}...")

    # Key summary metrics for each model
    print(f"\n  ── Key Metrics ──")
    print(f"  {'Metric':<25s}", end="")
    for label in labels:
        print(f" {label:>14s}", end="")
    print()
    print(f"  {'─'*25}", end="")
    for _ in labels:
        print(f" {'─'*14}", end="")
    print()

    key_metrics = [
        ("Final entropy", "logit_lens_entropy", lambda a: f"{a[-1]:.3f}"),
        ("Mean wasted work", "dilution_wasted_work", lambda a: f"{np.mean(a):.1%}"),
        ("Mean survival", "dilution_survival", lambda a: f"{np.mean(a):.3f}"),
        ("Norm growth", "residual_norms", lambda a: f"{a[-1]/(a[0]+1e-10):.1f}x"),
        ("MLP dominance", "mlp_attn_ratio", lambda a: f"{np.mean(a):.1%}"),
        ("Mean attn norm", "attn_norms", lambda a: f"{np.mean(a):.1f}"),
        ("Mean direction Δ", "direction_changes", lambda a: f"{np.mean(a):.4f}"),
        ("Mean entropy red.", "entropy_reduction", lambda a: f"{np.mean(a):.4f}"),
    ]

    for metric_name, signal, fmt_fn in key_metrics:
        print(f"  {metric_name:<25s}", end="")
        for data in all_data:
            arr = extract_signal(data, signal)
            if arr is not None:
                print(f" {fmt_fn(arr):>14s}", end="")
            else:
                print(f" {'N/A':>14s}", end="")
        print()

    # Pairwise diffs (consecutive pairs)
    for i in range(n - 1):
        a_label, b_label = labels[i], labels[i + 1]
        a_data, b_data = all_data[i], all_data[i + 1]

        print(f"\n  ── {a_label} → {b_label}: Signal Changes ──")
        print(f"  {'Signal':<30s} {'Mean Δ':>10s} {'Max Δ':>10s} {'Max Layer':>10s} {'Mean Rel':>10s}")
        print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

        diffs_for_sort = []
        for signal in SIGNALS:
            a_arr = extract_signal(a_data, signal)
            b_arr = extract_signal(b_data, signal)
            if a_arr is None or b_arr is None:
                continue
            d = pairwise_diff(a_arr, b_arr)
            diffs_for_sort.append((signal, d))

        diffs_for_sort.sort(key=lambda x: x[1]["mean_abs_diff"], reverse=True)
        for signal, d in diffs_for_sort:
            print(f"  {signal:<30s} {d['mean_abs_diff']:>10.4f} {d['max_abs_diff']:>10.4f} "
                  f"L{d['max_abs_layer']:>8} {d['mean_rel_diff']:>9.1%}")

    # Layer-type breakdown for hybrid architectures
    if layer_types and any(lt != "unknown" for lt in layer_types):
        print(f"\n  ── Layer-Type Breakdown ──")
        for signal_name in ["dilution_survival", "entropy_reduction",
                            "dilution_wasted_work", "attn_norms"]:
            print(f"\n  {signal_name}:")
            full_idx = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
            lin_idx = [i for i, lt in enumerate(layer_types) if lt == "linear_attention"]

            for label, data in zip(labels, all_data):
                arr = extract_signal(data, signal_name)
                if arr is None:
                    continue
                full_mean = np.mean([arr[i] for i in full_idx if i < len(arr)]) if full_idx else float('nan')
                lin_mean = np.mean([arr[i] for i in lin_idx if i < len(arr)]) if lin_idx else float('nan')
                all_mean = np.mean(arr)
                print(f"    [{label:>12s}] all={all_mean:>8.4f}  "
                      f"full_attn={full_mean:>8.4f}  linear={lin_mean:>8.4f}")


# ── Visualization ─────────────────────────────────────────────────

def visualize_nway(all_data: List[dict], labels: List[str], output_path: str):
    """Overlay visualization for N models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    layer_types = all_data[0].get("layer_types")

    signal_configs = [
        ("residual_norms", "Residual Norm"),
        ("residual_deltas", "Layer Delta"),
        ("logit_lens_entropy", "Logit Lens Entropy"),
        ("dilution_survival", "Dilution Survival"),
        ("dilution_wasted_work", "Wasted Work"),
        ("dilution_cumulative_drift", "Cumulative Drift"),
        ("mlp_attn_ratio", "MLP Dominance"),
        ("attn_norms", "Attention Norm"),
        ("entropy_reduction", "Entropy Reduction"),
        ("outlier_max_magnitude", "Max Activation"),
        ("outlier_gini", "Gini Coefficient"),
        ("direction_changes", "Direction Change"),
    ]

    n_plots = len(signal_configs)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    model_names = [d.get("model", l).split("/")[-1] for d, l in zip(all_data, labels)]
    fig.suptitle(f"Telemetry Comparison: {' vs '.join(model_names)}", fontsize=13, fontweight="bold")

    axes_flat = axes.flatten()

    for idx, (signal, title) in enumerate(signal_configs):
        ax = axes_flat[idx]

        has_data = False
        for i, (data, label) in enumerate(zip(all_data, labels)):
            arr = extract_signal(data, signal)
            if arr is None:
                continue
            has_data = True
            x = range(len(arr))
            ax.plot(x, arr, color=colors[i % len(colors)], linewidth=1.8,
                    label=label, alpha=0.85)

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        # Mark full attention layers
        if layer_types:
            for i, lt in enumerate(layer_types):
                if lt == "full_attention":
                    ax.axvline(x=i, color="green", alpha=0.08, linewidth=4)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Layer")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Visualization saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N-way Telemetry Comparison (base → instruct → reasoning)")
    parser.add_argument("--models", type=str, nargs="+",
                        help="Model names (HuggingFace), in order")
    parser.add_argument("--labels", type=str, nargs="+",
                        help="Short labels for each model (e.g. base instruct reasoning)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--prompt", type=str,
                        default="The theory of everything begins with the observation that",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--output-dir", type=str, default="data/telemetry")
    parser.add_argument("--from-json", type=str, nargs="+", default=None,
                        metavar="JSON_PATH",
                        help="Load from existing telemetry JSON files instead of running models")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        # Load from existing files
        all_data = []
        for path in args.from_json:
            print(f"  Loading {path}...")
            with open(path) as f:
                all_data.append(json.load(f))
        texts = ["(from JSON)"] * len(all_data)
        if not args.labels:
            args.labels = [f"model_{i}" for i in range(len(all_data))]
    else:
        if not args.models:
            parser.error("--models required (or use --from-json)")
        if not args.labels:
            args.labels = [m.split("/")[-1] for m in args.models]
        if len(args.labels) != len(args.models):
            parser.error(f"Got {len(args.models)} models but {len(args.labels)} labels")

        all_data = []
        texts = []

        for i, (model_name, label) in enumerate(zip(args.models, args.labels)):
            print(f"\n{'='*70}")
            print(f"  MODEL {i+1}/{len(args.models)}: {label.upper()} ({model_name})")
            print(f"{'='*70}")

            model, tokenizer = load_model(model_name, args.device, args.dtype)
            data, text = run_telemetry(
                model, tokenizer, args.prompt, args.max_tokens, args.device, label
            )

            # Save individual telemetry
            safe_name = model_name.replace("/", "-")
            json_path = out_dir / f"telemetry_{safe_name}_{label}.json"
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved: {json_path}")

            all_data.append(data)
            texts.append(text)

            free_model(model, tokenizer, args.device)

    # Report
    print_nway_report(all_data, args.labels, texts)

    # Visualization
    viz_name = "_vs_".join(args.labels)
    viz_path = out_dir / f"comparison_{viz_name}.png"
    visualize_nway(all_data, args.labels, str(viz_path))

    # Save combined diff report
    report = {
        "models": [d.get("model") for d in all_data],
        "labels": args.labels,
        "n_layers": all_data[0]["n_layers"],
        "layer_types": all_data[0].get("layer_types"),
        "pairwise_diffs": {},
    }
    for i in range(len(all_data) - 1):
        pair_key = f"{args.labels[i]}_vs_{args.labels[i+1]}"
        pair_diffs = {}
        for signal in SIGNALS:
            a = extract_signal(all_data[i], signal)
            b = extract_signal(all_data[i + 1], signal)
            if a is not None and b is not None:
                pair_diffs[signal] = pairwise_diff(a, b)
        report["pairwise_diffs"][pair_key] = pair_diffs

    report_path = out_dir / f"diff_{viz_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Diff report saved → {report_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
