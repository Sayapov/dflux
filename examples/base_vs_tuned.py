#!/usr/bin/env python3
"""
Base vs Tuned — telemetry comparison between a base model and its tuned variant.

Runs identical prompts through both models and produces:
  1. Per-field numerical diff (which signals changed, by how much)
  2. Per-layer breakdown (where does tuning have the most effect?)
  3. Visualization comparing all telemetry signals side-by-side
  4. Dilution/entropy/norm comparison overlay

Usage:
    # Qwen3.5-9B base vs instruct (BF16 for memory)
    python examples/base_vs_tuned.py \
        --base Qwen/Qwen3.5-9B-Base \
        --tuned Qwen/Qwen3.5-9B \
        --device mps --dtype bfloat16

    # GPT-2 (small, fast, float32)
    python examples/base_vs_tuned.py \
        --base gpt2 --tuned gpt2  # sanity check: should be identical

    # From existing telemetry JSONs
    python examples/base_vs_tuned.py \
        --compare-json data/telemetry/telemetry_base.json data/telemetry/telemetry_tuned.json
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dflux.multiscale_telemetry import MultiScaleTelemetry, TelemetryConfig


def _detect_nested_keys(model_name: str) -> bool:
    """Peek at checkpoint key format without loading weights.

    Returns True if keys contain 'language_model.' (official Qwen3.5),
    False if flat (Unsloth/PEFT fine-tunes).
    """
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
        return True  # default: assume nested


def _load_qwen35(model_name: str, config, torch_dtype, device: str):
    """Qwen3.5 loading: promote config, detect key format, load/remap."""
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

    needs_remap = _detect_nested_keys(model_name)

    if not needs_remap:
        print(f"  Detected flat key format, using from_pretrained...")
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_name, config=config, dtype=torch_dtype,
            trust_remote_code=True,
        )
        return model.to(device)

    # Nested key format — manual load + remap
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

    # Detect model type — some architectures (Qwen3.5, Gemma3) have nested
    # configs where AutoModelForCausalLM passes the wrong config level.
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_5":
        model = _load_qwen35(model_name, config, torch_dtype, device)

        model = model.to(device)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch_dtype, trust_remote_code=True,
            ).to(device)
        except AttributeError:
            # Fallback: try with dtype= instead of torch_dtype= (newer API)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch_dtype, trust_remote_code=True,
            ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {n_params:.0f}M params on {device} ({torch_dtype})")
    return model, tokenizer


def run_telemetry(model, tokenizer, prompt: str, max_tokens: int, device: str, label: str):
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

    # Save to temp file and reload for clean comparison
    tmp = f"/tmp/bvt_{label.replace(' ', '_')}.json"
    telem.save(tmp)
    with open(tmp) as f:
        data = json.load(f)

    telem.detach()
    return data, text


def compute_signal_diff(base_data: dict, tuned_data: dict) -> dict:
    """Compute per-signal, per-layer differences between base and tuned telemetry."""

    base_agg = base_data["aggregate"]
    tuned_agg = tuned_data["aggregate"]
    n_layers = base_data["n_layers"]

    # Signals to compare (all per-layer mean values from aggregate)
    signals = [
        "residual_norms", "residual_deltas", "direction_changes",
        "logit_lens_entropy", "logit_lens_top1_prob",
        "mlp_norms", "attn_norms", "mlp_attn_ratio",
        "mlp_dead_frac", "mlp_outlier_ratio",
        "entropy_reduction",
        "outlier_max_magnitude", "outlier_gini",
        "dilution_survival", "dilution_energy_frac",
        "dilution_wasted_work", "dilution_cumulative_drift",
    ]

    diffs = {}
    for signal in signals:
        base_key = f"{signal}_mean"
        tuned_key = f"{signal}_mean"

        base_vals = base_agg.get(base_key)
        tuned_vals = tuned_agg.get(tuned_key)

        if base_vals is None or tuned_vals is None:
            continue

        # Handle different lengths (different number of layers)
        min_len = min(len(base_vals), len(tuned_vals))
        base_arr = np.array(base_vals[:min_len])
        tuned_arr = np.array(tuned_vals[:min_len])

        abs_diff = np.abs(tuned_arr - base_arr)
        rel_diff = abs_diff / (np.abs(base_arr) + 1e-10)

        diffs[signal] = {
            "base": base_arr.tolist(),
            "tuned": tuned_arr.tolist(),
            "abs_diff": abs_diff.tolist(),
            "rel_diff": rel_diff.tolist(),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "max_abs_diff": float(np.max(abs_diff)),
            "max_abs_layer": int(np.argmax(abs_diff)),
            "mean_rel_diff": float(np.mean(rel_diff)),
        }

    return diffs


def print_comparison_report(base_data: dict, tuned_data: dict,
                            base_text: str, tuned_text: str,
                            diffs: dict):
    """Print human-readable comparison."""
    base_name = base_data.get("model", "base")
    tuned_name = tuned_data.get("model", "tuned")

    print(f"\n{'='*70}")
    print(f"  BASE vs TUNED TELEMETRY COMPARISON")
    print(f"{'='*70}")
    print(f"  Base:  {base_name} ({base_data['n_layers']} layers)")
    print(f"  Tuned: {tuned_name} ({tuned_data['n_layers']} layers)")

    # Layer type info
    base_lt = base_data.get("layer_types")
    tuned_lt = tuned_data.get("layer_types")
    if base_lt:
        n_full = sum(1 for lt in base_lt if lt == "full_attention")
        n_lin = sum(1 for lt in base_lt if lt == "linear_attention")
        print(f"  Architecture: hybrid ({n_full} full attention, {n_lin} linear attention)")

    # Text comparison
    print(f"\n  ── Generated Text ──")
    if base_text == tuned_text:
        print(f"  IDENTICAL output")
    else:
        print(f"  Base:  {base_text[:100]}...")
        print(f"  Tuned: {tuned_text[:100]}...")

    # Signal comparison sorted by impact
    print(f"\n  ── Signal Changes (sorted by magnitude) ──")
    print(f"  {'Signal':<30} {'Mean Δ':>10} {'Max Δ':>10} {'Max Layer':>10} {'Mean Rel':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    sorted_signals = sorted(diffs.items(), key=lambda x: x[1]["mean_abs_diff"], reverse=True)
    for signal, info in sorted_signals:
        print(f"  {signal:<30} {info['mean_abs_diff']:>10.4f} {info['max_abs_diff']:>10.4f} "
              f"L{info['max_abs_layer']:>8} {info['mean_rel_diff']:>9.1%}")

    # Per-layer impact (sum of all signal diffs at each layer)
    n_layers = base_data["n_layers"]
    layer_impact = np.zeros(n_layers)
    for signal, info in diffs.items():
        arr = np.array(info["rel_diff"])
        layer_impact[:len(arr)] += arr

    print(f"\n  ── Per-Layer Impact (sum of relative changes) ──")
    top_layers = np.argsort(layer_impact)[::-1][:10]
    for layer in top_layers:
        if layer_impact[layer] > 0:
            bar = "█" * int(layer_impact[layer] / max(layer_impact) * 40)
            lt_label = ""
            if base_lt and layer < len(base_lt):
                lt_label = f" [{base_lt[layer][:3]}]"
            print(f"  L{layer:>2}{lt_label}: {layer_impact[layer]:>8.3f} {bar}")

    # Key insights
    print(f"\n  ── Key Differences ──")

    # Dilution
    if "dilution_wasted_work" in diffs:
        base_waste = np.mean(diffs["dilution_wasted_work"]["base"])
        tuned_waste = np.mean(diffs["dilution_wasted_work"]["tuned"])
        print(f"  Mean wasted work: {base_waste:.1%} (base) → {tuned_waste:.1%} (tuned)")

    # Entropy
    if "logit_lens_entropy" in diffs:
        base_ent = diffs["logit_lens_entropy"]["base"][-1]
        tuned_ent = diffs["logit_lens_entropy"]["tuned"][-1]
        print(f"  Final layer entropy: {base_ent:.3f} (base) → {tuned_ent:.3f} (tuned)")

    # Norm growth
    if "residual_norms" in diffs:
        base_growth = diffs["residual_norms"]["base"][-1] / (diffs["residual_norms"]["base"][0] + 1e-10)
        tuned_growth = diffs["residual_norms"]["tuned"][-1] / (diffs["residual_norms"]["tuned"][0] + 1e-10)
        print(f"  Norm growth: {base_growth:.1f}x (base) → {tuned_growth:.1f}x (tuned)")

    # MLP dominance
    if "mlp_attn_ratio" in diffs:
        base_mlp = np.mean(diffs["mlp_attn_ratio"]["base"])
        tuned_mlp = np.mean(diffs["mlp_attn_ratio"]["tuned"])
        print(f"  MLP dominance: {base_mlp:.1%} (base) → {tuned_mlp:.1%} (tuned)")


def visualize_comparison(base_data: dict, tuned_data: dict, diffs: dict, output_path: str):
    """Side-by-side visualization of base vs tuned telemetry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_name = base_data.get("model", "base").split("/")[-1]
    tuned_name = tuned_data.get("model", "tuned").split("/")[-1]
    base_lt = base_data.get("layer_types")

    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f"Telemetry Comparison: {base_name} vs {tuned_name}", fontsize=14, fontweight="bold")

    # Color scheme
    base_color = "#2196F3"
    tuned_color = "#FF5722"

    signal_configs = [
        ("residual_norms", "Residual Norm", axes[0, 0]),
        ("residual_deltas", "Layer Delta", axes[0, 1]),
        ("logit_lens_entropy", "Entropy", axes[0, 2]),
        ("dilution_survival", "Dilution Survival", axes[1, 0]),
        ("dilution_wasted_work", "Wasted Work", axes[1, 1]),
        ("dilution_cumulative_drift", "Cumulative Drift", axes[1, 2]),
        ("mlp_attn_ratio", "MLP Dominance", axes[2, 0]),
        ("mlp_norms", "MLP Norm", axes[2, 1]),
        ("attn_norms", "Attention Norm", axes[2, 2]),
        ("outlier_max_magnitude", "Max Activation", axes[3, 0]),
        ("outlier_gini", "Gini Coefficient", axes[3, 1]),
        ("direction_changes", "Direction Change", axes[3, 2]),
    ]

    for signal, title, ax in signal_configs:
        if signal not in diffs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        base_vals = diffs[signal]["base"]
        tuned_vals = diffs[signal]["tuned"]
        x = range(len(base_vals))

        ax.plot(x, base_vals, color=base_color, linewidth=1.5, label="Base", alpha=0.8)
        ax.plot(x, tuned_vals, color=tuned_color, linewidth=1.5, label="Tuned", alpha=0.8)

        # Shade the difference
        ax.fill_between(x, base_vals, tuned_vals, alpha=0.15, color="gray")

        # Mark layer types on x-axis if hybrid
        if base_lt:
            for i, lt in enumerate(base_lt):
                if i < len(base_vals) and lt == "full_attention":
                    ax.axvline(x=i, color="green", alpha=0.1, linewidth=3)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Layer")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Visualization saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Base vs Tuned Telemetry Comparison")
    parser.add_argument("--base", type=str, help="Base model name (HuggingFace)")
    parser.add_argument("--tuned", type=str, help="Tuned model name (HuggingFace)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, mps")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (bfloat16 for large models)")
    parser.add_argument("--prompt", type=str,
                        default="The theory of everything begins with the observation that",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=48, help="Max new tokens")
    parser.add_argument("--output-dir", type=str, default="data/telemetry",
                        help="Where to save outputs")
    parser.add_argument("--compare-json", type=str, nargs=2, default=None,
                        metavar=("BASE_JSON", "TUNED_JSON"),
                        help="Compare two existing telemetry JSON files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_json:
        # Load from existing JSON files
        print(f"  Loading {args.compare_json[0]}...")
        with open(args.compare_json[0]) as f:
            base_data = json.load(f)
        print(f"  Loading {args.compare_json[1]}...")
        with open(args.compare_json[1]) as f:
            tuned_data = json.load(f)
        base_text = "(from JSON)"
        tuned_text = "(from JSON)"
    else:
        if not args.base or not args.tuned:
            parser.error("--base and --tuned are required (or use --compare-json)")

        # Run base model
        print(f"\n{'='*70}")
        print(f"  BASE MODEL")
        print(f"{'='*70}")
        base_model, base_tok = load_model(args.base, args.device, args.dtype)
        base_data, base_text = run_telemetry(
            base_model, base_tok, args.prompt, args.max_tokens, args.device, "base"
        )

        # Save base telemetry
        safe_base = args.base.replace("/", "-")
        base_json = out_dir / f"telemetry_{safe_base}_base.json"
        with open(base_json, "w") as f:
            json.dump(base_data, f, indent=2)
        print(f"  Saved: {base_json}")

        # Free memory
        del base_model, base_tok
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()

        # Run tuned model
        print(f"\n{'='*70}")
        print(f"  TUNED MODEL")
        print(f"{'='*70}")
        tuned_model, tuned_tok = load_model(args.tuned, args.device, args.dtype)
        tuned_data, tuned_text = run_telemetry(
            tuned_model, tuned_tok, args.prompt, args.max_tokens, args.device, "tuned"
        )

        # Save tuned telemetry
        safe_tuned = args.tuned.replace("/", "-")
        tuned_json = out_dir / f"telemetry_{safe_tuned}_tuned.json"
        with open(tuned_json, "w") as f:
            json.dump(tuned_data, f, indent=2)
        print(f"  Saved: {tuned_json}")

        del tuned_model, tuned_tok
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()

    # Verify same architecture
    if base_data["n_layers"] != tuned_data["n_layers"]:
        print(f"\n  WARNING: Layer count mismatch ({base_data['n_layers']} vs {tuned_data['n_layers']})")
        print(f"  Comparison will use the shorter model's layer count.")

    # Compare
    diffs = compute_signal_diff(base_data, tuned_data)
    print_comparison_report(base_data, tuned_data, base_text, tuned_text, diffs)

    # Visualize
    base_name = base_data.get("model", "base").split("/")[-1].replace("/", "-")
    tuned_name = tuned_data.get("model", "tuned").split("/")[-1].replace("/", "-")
    viz_path = out_dir / f"comparison_{base_name}_vs_{tuned_name}.png"
    visualize_comparison(base_data, tuned_data, diffs, str(viz_path))

    # Save diff report
    report_path = out_dir / f"diff_{base_name}_vs_{tuned_name}.json"
    report = {
        "base_model": base_data.get("model"),
        "tuned_model": tuned_data.get("model"),
        "n_layers": base_data["n_layers"],
        "layer_types": base_data.get("layer_types"),
        "text_match": base_text == tuned_text,
        "signal_diffs": {k: {kk: vv for kk, vv in v.items()
                             if kk not in ("base", "tuned")}
                         for k, v in diffs.items()},
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Diff report saved → {report_path}")


if __name__ == "__main__":
    main()
