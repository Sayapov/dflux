#!/usr/bin/env python3
"""
Distillation Governor — can we make a base model act like a reasoning model
by scaling attention outputs to match the reasoning model's telemetry profile?

The experiment:
  1. Load telemetry from base model and reasoning-distilled model
  2. Compute per-layer scale targets from the telemetry diff
  3. Load the BASE model
  4. Run generation with the distillation governor active
  5. Run telemetry on the governed output to see if signals shifted
  6. Compare: base (ungoverned) vs base (governed) vs reasoning (reference)

Usage:
    # Using existing telemetry JSONs from three_way_comparison.py
    python examples/distillation_governor.py \
        --base-model Qwen/Qwen3.5-9B-Base \
        --base-telemetry data/telemetry/telemetry_Qwen-Qwen3.5-9B-Base_base.json \
        --target-telemetry data/telemetry/telemetry_Jackrong-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled_reasoning.json \
        --device mps --dtype bfloat16

    # Quick test on GPT-2 (use two copies of same telemetry for sanity check)
    python examples/distillation_governor.py \
        --base-model gpt2 \
        --base-telemetry data/telemetry/telemetry_gpt2.json \
        --target-telemetry data/telemetry/telemetry_gpt2.json \
        --device mps

    # With blend factor (0.5 = half effect, 2.0 = overshoot)
    python examples/distillation_governor.py \
        --base-model Qwen/Qwen3.5-9B-Base \
        --base-telemetry data/telemetry/base.json \
        --target-telemetry data/telemetry/reasoning.json \
        --blend 1.5 --signal entropy_reduction \
        --device mps --dtype bfloat16
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
from dflux.live_governor import LiveGovernor


# ── Model loading (shared with other examples) ───────────────────

def _detect_nested_keys(model_name: str) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx_path) as f:
            keys = json.load(f)["weight_map"].keys()
        return any("language_model." in k for k in keys)
    except Exception:
        pass
    try:
        from safetensors import safe_open
        path = hf_hub_download(model_name, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            keys = f.keys()
        return any("language_model." in k for k in keys)
    except Exception:
        return True


def _load_qwen35(model_name, config, torch_dtype, device):
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

    if not _detect_nested_keys(model_name):
        print(f"  Detected flat key format, using from_pretrained...")
        return Qwen3_5ForCausalLM.from_pretrained(
            model_name, config=config, dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)

    print(f"  Detected nested keys, using manual remap...")
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
    print(f"  Remapped {loaded} weight tensors")
    del full_sd
    gc.collect()
    return model.to(device)


def load_model(model_name, device, dtype="float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    print(f"\n  Loading {model_name} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if getattr(config, "model_type", "") == "qwen3_5":
        model = _load_qwen35(model_name, config, torch_dtype, device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, trust_remote_code=True,
        ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {n_params:.0f}M params on {device}")
    return model, tokenizer


# ── Telemetry capture ─────────────────────────────────────────────

def capture_telemetry(model, tokenizer, prompt, max_tokens, device):
    cfg = TelemetryConfig(
        logit_lens=True, logit_lens_top_k=5, cross_layer=False,
        mlp_internals=True, entropy_cascade=True, outlier_detection=True,
    )
    telem = MultiScaleTelemetry.from_model(model, tokenizer, cfg=cfg)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(42)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    tmp = "/tmp/distill_telem.json"
    telem.save(tmp)
    with open(tmp) as f:
        data = json.load(f)
    telem.detach()
    return data, text


# ── Analysis ──────────────────────────────────────────────────────

def print_scale_profile(gov, label=""):
    """Print the governor's static scale profile."""
    print(f"\n  ── Scale Profile{' (' + label + ')' if label else ''} ──")
    for i in range(gov.n_layers):
        s = gov._scales[i].item()
        if abs(s - 1.0) > 0.001:
            lt = ""
            if gov.layer_types and i < len(gov.layer_types):
                lt = f" [{gov.layer_types[i][:3]}]"
            direction = "↑" if s > 1.0 else "↓"
            bar_len = int(abs(s - 1.0) * 40)
            bar = "█" * min(40, bar_len)
            print(f"  L{i:>2}{lt}: {s:.3f} {direction} {bar}")


def compare_signals(base_data, governed_data, target_data, signals, layer_types=None):
    """Compare key signals across base, governed, and target."""
    print(f"\n  ── Signal Comparison ──")
    print(f"  {'Signal':<28s} {'base':>10s} {'governed':>10s} {'target':>10s} {'gap_before':>12s} {'gap_after':>12s}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")

    for signal in signals:
        bv = base_data.get("aggregate", {}).get(f"{signal}_mean")
        gv = governed_data.get("aggregate", {}).get(f"{signal}_mean")
        tv = target_data.get("aggregate", {}).get(f"{signal}_mean")
        if bv is None or tv is None:
            continue

        b_mean = np.mean(bv)
        t_mean = np.mean(tv)
        g_mean = np.mean(gv) if gv else b_mean

        gap_before = abs(b_mean - t_mean)
        gap_after = abs(g_mean - t_mean)
        improved = gap_after < gap_before
        marker = "CLOSER" if improved else "further"

        print(f"  {signal:<28s} {b_mean:>10.4f} {g_mean:>10.4f} {t_mean:>10.4f} "
              f"{gap_before:>12.4f} {gap_after:>12.4f} {marker}")

    # Layer-type breakdown
    if layer_types and any(lt != "unknown" for lt in layer_types):
        full_idx = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
        lin_idx = [i for i, lt in enumerate(layer_types) if lt == "linear_attention"]

        for sig in ["entropy_reduction", "dilution_survival"]:
            bv = base_data.get("aggregate", {}).get(f"{sig}_mean")
            gv = governed_data.get("aggregate", {}).get(f"{sig}_mean")
            tv = target_data.get("aggregate", {}).get(f"{sig}_mean")
            if bv is None or tv is None:
                continue

            print(f"\n  {sig} by layer type:")
            for type_name, indices in [("full_attn", full_idx), ("linear", lin_idx)]:
                b = np.mean([bv[i] for i in indices if i < len(bv)])
                t = np.mean([tv[i] for i in indices if i < len(tv)])
                g = np.mean([gv[i] for i in indices if gv and i < len(gv)]) if gv else b
                gap_b = abs(b - t)
                gap_a = abs(g - t)
                marker = "CLOSER" if gap_a < gap_b else "further"
                print(f"    {type_name:<10s}: base={b:.4f} governed={g:.4f} target={t:.4f} "
                      f"gap {gap_b:.4f}→{gap_a:.4f} {marker}")


# ── Main ──────────────────────────────────────────────────────────

PROMPTS = [
    "The theory of everything begins with the observation that",
    "The key to understanding deep learning is",
    "In the year 2050, artificial intelligence will",
    "The most important scientific discovery of the 21st century was",
]


def main():
    parser = argparse.ArgumentParser(description="Distillation Governor Experiment")
    parser.add_argument("--base-model", required=True, help="Base model to govern")
    parser.add_argument("--base-telemetry", required=True, help="Base model telemetry JSON")
    parser.add_argument("--target-telemetry", required=True, help="Target model telemetry JSON")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--signal", default="entropy_reduction",
                        help="Signal to use for scale computation")
    parser.add_argument("--strategy", default="ratio", choices=["ratio", "delta"])
    parser.add_argument("--blend", type=float, default=1.0,
                        help="Blend factor: 0=none, 1=full, >1=overshoot")
    parser.add_argument("--cap", type=float, default=2.0,
                        help="Max scale value (safety clamp)")
    parser.add_argument("--layer-type-bias", default=None,
                        choices=["full_attention", "linear_attention"],
                        help="Only scale this layer type")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--prompt-idx", type=int, default=None,
                        help="Use only this prompt index (default: all)")
    parser.add_argument("--output-dir", default="data/telemetry")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load telemetry references
    print(f"  Loading base telemetry: {args.base_telemetry}")
    with open(args.base_telemetry) as f:
        base_telem = json.load(f)
    print(f"  Loading target telemetry: {args.target_telemetry}")
    with open(args.target_telemetry) as f:
        target_telem = json.load(f)

    layer_types = base_telem.get("layer_types")
    n_layers = base_telem["n_layers"]

    # Show what the telemetry diff looks like
    print(f"\n{'='*70}")
    print(f"  TELEMETRY DIFF ({args.signal})")
    print(f"{'='*70}")

    base_sig = base_telem["aggregate"].get(f"{args.signal}_mean", [])
    target_sig = target_telem["aggregate"].get(f"{args.signal}_mean", [])
    if base_sig and target_sig:
        for i in range(min(len(base_sig), len(target_sig))):
            lt = f" [{layer_types[i][:3]}]" if layer_types and i < len(layer_types) else ""
            ratio = target_sig[i] / base_sig[i] if abs(base_sig[i]) > 1e-10 else 1.0
            delta = target_sig[i] - base_sig[i]
            print(f"  L{i:>2}{lt}: base={base_sig[i]:>8.4f}  target={target_sig[i]:>8.4f}  "
                  f"ratio={ratio:>6.2f}  delta={delta:>+8.4f}")

    # Load base model
    print(f"\n{'='*70}")
    print(f"  LOADING BASE MODEL")
    print(f"{'='*70}")
    model, tokenizer = load_model(args.base_model, args.device, args.dtype)

    prompts = PROMPTS if args.prompt_idx is None else [PROMPTS[args.prompt_idx]]

    # Phase 1: Baseline (ungoverned)
    print(f"\n{'='*70}")
    print(f"  PHASE 1: BASELINE (ungoverned)")
    print(f"{'='*70}")

    baseline_results = []
    for prompt in prompts:
        data, text = capture_telemetry(model, tokenizer, prompt, args.max_tokens, args.device)
        print(f"\n  \"{prompt}\"")
        print(f"  → \"{text[len(prompt):len(prompt)+120]}\"")
        baseline_results.append({"prompt": prompt, "text": text, "telemetry": data})

    # Phase 2: Governed (distillation)
    print(f"\n{'='*70}")
    print(f"  PHASE 2: DISTILLATION GOVERNOR")
    print(f"  Signal: {args.signal}, Strategy: {args.strategy}, Blend: {args.blend}")
    print(f"{'='*70}")

    gov = LiveGovernor.from_telemetry_diff(
        model, tokenizer,
        base_telem, target_telem,
        signal=args.signal,
        strategy=args.strategy,
        blend=args.blend,
        cap=args.cap,
        layer_type_bias=args.layer_type_bias,
    )

    print_scale_profile(gov, f"{args.signal} {args.strategy} blend={args.blend}")

    governed_results = []
    for prompt in prompts:
        gov.telem.snapshots.clear()
        gov.scale_history.clear()
        gov.interventions.clear()

        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        torch.manual_seed(42)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=args.max_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n  \"{prompt}\"")
        print(f"  → \"{text[len(prompt):len(prompt)+120]}\"")

        # Capture governed telemetry (the telem hooks are active inside the governor)
        tmp = "/tmp/distill_governed.json"
        gov.telem.save(tmp)
        with open(tmp) as f:
            gov_telem_data = json.load(f)

        governed_results.append({
            "prompt": prompt, "text": text, "telemetry": gov_telem_data,
        })

    # Capture scale profile BEFORE detaching (detach resets scales to 1.0)
    saved_scale_profile = {i: gov._scales[i].item() if i in gov._scales else 1.0
                           for i in range(n_layers)}

    gov.detach()

    # Phase 3: Compare
    print(f"\n{'='*70}")
    print(f"  PHASE 3: COMPARISON")
    print(f"{'='*70}")

    key_signals = [
        "entropy_reduction", "dilution_survival", "dilution_wasted_work",
        "logit_lens_entropy", "attn_norms", "residual_norms",
    ]

    for i, (base_r, gov_r) in enumerate(zip(baseline_results, governed_results)):
        print(f"\n  ── Prompt {i}: \"{base_r['prompt'][:50]}...\" ──")
        base_text = base_r["text"][len(base_r["prompt"]):]
        gov_text = gov_r["text"][len(gov_r["prompt"]):]
        if base_text == gov_text:
            print(f"  Text: IDENTICAL")
        else:
            print(f"  Base:     \"{base_text[:100]}\"")
            print(f"  Governed: \"{gov_text[:100]}\"")

        compare_signals(
            base_r["telemetry"], gov_r["telemetry"], target_telem,
            key_signals, layer_types,
        )

    # Summary: how much closer did we get?
    print(f"\n{'='*70}")
    print(f"  SUMMARY: DID THE BASE MODEL MOVE TOWARD THE TARGET?")
    print(f"{'='*70}")

    for sig in key_signals:
        tv = target_telem.get("aggregate", {}).get(f"{sig}_mean")
        if tv is None:
            continue
        t_mean = np.mean(tv)

        gaps_before = []
        gaps_after = []
        for base_r, gov_r in zip(baseline_results, governed_results):
            bv = base_r["telemetry"].get("aggregate", {}).get(f"{sig}_mean")
            gv = gov_r["telemetry"].get("aggregate", {}).get(f"{sig}_mean")
            if bv and gv:
                gaps_before.append(abs(np.mean(bv) - t_mean))
                gaps_after.append(abs(np.mean(gv) - t_mean))

        if gaps_before and gaps_after:
            mean_before = np.mean(gaps_before)
            mean_after = np.mean(gaps_after)
            pct = (mean_after - mean_before) / mean_before * 100 if mean_before > 0 else 0
            marker = "CLOSER" if mean_after < mean_before else "FURTHER"
            print(f"  {sig:<28s}: gap {mean_before:.4f} → {mean_after:.4f} ({pct:+.1f}%) {marker}")

    # Save results
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            return None if obj != obj else round(obj, 6)
        elif isinstance(obj, np.floating):
            return round(float(obj), 6)
        return obj

    save_data = {
        "config": {
            "base_model": args.base_model,
            "signal": args.signal,
            "strategy": args.strategy,
            "blend": args.blend,
            "cap": args.cap,
            "layer_type_bias": args.layer_type_bias,
        },
        "scale_profile": saved_scale_profile,
        "baseline": sanitize([{
            "prompt": r["prompt"],
            "text": r["text"],
        } for r in baseline_results]),
        "governed": sanitize([{
            "prompt": r["prompt"],
            "text": r["text"],
        } for r in governed_results]),
    }

    safe_name = args.base_model.replace("/", "-")
    save_path = out_dir / f"distillation_{safe_name}_{args.signal}.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved → {save_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
