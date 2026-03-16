#!/usr/bin/env python3
"""
Live Governor Demo: Real-time inference-time head scaling driven by telemetry.
==============================================================================

Runs four governor presets on a model and compares governed vs ungoverned
generation.  For each preset the script:

  1. Generates text WITHOUT the governor (baseline)
  2. Generates text WITH the governor active
  3. Prints the governor report (interventions, scale adjustments)
  4. Compares output quality (perplexity proxy via final entropy)

The governor modifies attention output projections *between* tokens based
on live telemetry — no weight mutation, fully reversible.

Usage:
    python examples/live_governor.py
    python examples/live_governor.py --model gpt2-medium
    python examples/live_governor.py --model gpt2 --preset survival --tokens 64
    python examples/live_governor.py --model gpt2 --preset all --device mps
"""

import sys, os, json, argparse, time

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux import MultiScaleTelemetry, TelemetryConfig
from dflux.live_governor import LiveGovernor, GovernorRule


def get_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PROMPTS = [
    "The key to understanding deep learning is",
    "In the year 2050, artificial intelligence will",
    "The most important scientific discovery of the 21st century was",
]


def generate(model, tokenizer, prompt, max_tokens=32, do_sample=False):
    """Generate text and return (output_ids, generated_text)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output, text


def run_baseline(model, tokenizer, prompts, max_tokens):
    """Run telemetry without governor to get baseline."""
    cfg = TelemetryConfig(
        logit_lens=True,
        logit_lens_top_k=5,
        cross_layer=False,
        mlp_internals=True,
        entropy_cascade=True,
        outlier_detection=True,
    )
    telem = MultiScaleTelemetry.from_model(model, tokenizer, cfg=cfg)
    results = []
    for prompt in prompts:
        telem.snapshots.clear()
        _, text = generate(model, tokenizer, prompt, max_tokens)
        # Grab final entropy from last snapshot
        final_entropy = None
        if telem.snapshots:
            last = telem.snapshots[-1]
            if hasattr(last, "entropy_cascade") and last.entropy_cascade:
                final_entropy = last.entropy_cascade[-1]
        results.append({
            "prompt": prompt,
            "text": text,
            "final_entropy": final_entropy,
            "n_snapshots": len(telem.snapshots),
        })
    telem.detach()
    return results


def run_governed(model, tokenizer, prompts, max_tokens, preset, **kwargs):
    """Run generation with a governor preset active."""
    factory = {
        "entropy": LiveGovernor.entropy_governor,
        "dominance": LiveGovernor.dominance_damper,
        "survival": LiveGovernor.survival_amplifier,
        "hybrid": LiveGovernor.hybrid_governor,
    }
    if preset not in factory:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(factory.keys())}")

    gov = factory[preset](model, tokenizer, **kwargs)
    results = []
    for prompt in prompts:
        gov.telem.snapshots.clear()
        gov.interventions.clear()
        gov.scale_history.clear()
        gov.reset_scales()

        _, text = generate(model, tokenizer, prompt, max_tokens)

        final_entropy = None
        if gov.telem.snapshots:
            last = gov.telem.snapshots[-1]
            if hasattr(last, "entropy_cascade") and last.entropy_cascade:
                final_entropy = last.entropy_cascade[-1]

        results.append({
            "prompt": prompt,
            "text": text,
            "final_entropy": final_entropy,
            "n_snapshots": len(gov.telem.snapshots),
            "report": gov.report(),
        })

    gov.detach()
    return results


def print_comparison(baseline_results, governed_results, preset_name):
    """Print side-by-side comparison."""
    print(f"\n{'─' * 70}")
    print(f"  PRESET: {preset_name.upper()}")
    print(f"{'─' * 70}")

    for i, (base, gov) in enumerate(zip(baseline_results, governed_results)):
        print(f"\n  Prompt: \"{base['prompt']}\"")
        print(f"  ── Baseline ──")
        print(f"    Text: \"{base['text'][:120]}\"")
        if base["final_entropy"] is not None:
            print(f"    Final entropy: {base['final_entropy']:.4f}")

        print(f"  ── Governed ──")
        print(f"    Text: \"{gov['text'][:120]}\"")
        if gov["final_entropy"] is not None:
            print(f"    Final entropy: {gov['final_entropy']:.4f}")

        report = gov.get("report", {})
        n_int = report.get("total_interventions", 0)
        ipt = report.get("interventions_per_token", 0)
        print(f"    Interventions: {n_int} total ({ipt:.1f}/token)")

        # Show top active layers
        top = report.get("top_layers", [])[:5]
        if top:
            layer_str = ", ".join(
                f"L{e['layer']}({e['mean_scale']:.2f})"
                for e in top
            )
            print(f"    Top layers: {layer_str}")

    # Summary entropy comparison
    base_entropies = [r["final_entropy"] for r in baseline_results if r["final_entropy"] is not None]
    gov_entropies = [r["final_entropy"] for r in governed_results if r["final_entropy"] is not None]
    if base_entropies and gov_entropies:
        base_mean = sum(base_entropies) / len(base_entropies)
        gov_mean = sum(gov_entropies) / len(gov_entropies)
        delta = gov_mean - base_mean
        pct = delta / base_mean * 100 if base_mean != 0 else 0
        print(f"\n  Mean final entropy: {base_mean:.4f} → {gov_mean:.4f} ({delta:+.4f}, {pct:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Live Governor Demo")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--tokens", type=int, default=32,
                        help="Max new tokens to generate")
    parser.add_argument("--preset", default="all",
                        choices=["entropy", "dominance", "survival", "hybrid", "all"],
                        help="Governor preset to run (default: all)")
    parser.add_argument("--mode", default="reactive",
                        choices=["reactive", "adaptive"],
                        help="Governor mode")
    args = parser.parse_args()

    device = get_device(args.device)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    print("=" * 70)
    print("  LIVE GOVERNOR DEMO")
    print("=" * 70)
    print(f"  Model:   {args.model}")
    print(f"  Device:  {device}")
    print(f"  Dtype:   {args.dtype}")
    print(f"  Tokens:  {args.tokens}")
    print(f"  Mode:    {args.mode}")
    print(f"  Preset:  {args.preset}")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype)
    model = model.to(device)
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params on {device}")

    # ── Baseline (ungoverned) ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  PHASE 1: BASELINE (no governor)")
    print(f"{'=' * 70}")

    t0 = time.time()
    baseline = run_baseline(model, tokenizer, PROMPTS, args.tokens)
    baseline_time = time.time() - t0

    for r in baseline:
        print(f"\n  \"{r['prompt']}\"")
        print(f"  → \"{r['text'][:120]}\"")
        if r["final_entropy"] is not None:
            print(f"  Final entropy: {r['final_entropy']:.4f}")
    print(f"\n  Baseline time: {baseline_time:.1f}s")

    # ── Governed runs ─────────────────────────────────────────
    presets = ["entropy", "dominance", "survival", "hybrid"] if args.preset == "all" else [args.preset]

    all_results = {}
    for preset in presets:
        print(f"\n{'=' * 70}")
        print(f"  PHASE 2: GOVERNED ({preset.upper()})")
        print(f"{'=' * 70}")

        t0 = time.time()
        governed = run_governed(
            model, tokenizer, PROMPTS, args.tokens,
            preset=preset, mode=args.mode,
        )
        gov_time = time.time() - t0

        print_comparison(baseline, governed, preset)
        print(f"\n  Governed time: {gov_time:.1f}s (overhead: {gov_time - baseline_time:+.1f}s)")

        all_results[preset] = governed

    # ── Save results ──────────────────────────────────────────
    output_dir = os.path.dirname(os.path.abspath(__file__))

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            return round(obj, 6)
        return obj

    save_data = {
        "config": {
            "model": args.model,
            "device": device,
            "dtype": args.dtype,
            "tokens": args.tokens,
            "mode": args.mode,
        },
        "baseline": sanitize(baseline),
        "governed": {k: sanitize(v) for k, v in all_results.items()},
    }

    data_path = os.path.join(output_dir, f"live_governor_{args.model.replace('/', '-')}.json")
    with open(data_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {data_path}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for preset, results in all_results.items():
        total_interventions = sum(r.get("report", {}).get("total_interventions", 0) for r in results)
        gov_entropies = [r["final_entropy"] for r in results if r["final_entropy"] is not None]
        base_entropies = [r["final_entropy"] for r in baseline if r["final_entropy"] is not None]
        if gov_entropies and base_entropies:
            ge = sum(gov_entropies) / len(gov_entropies)
            be = sum(base_entropies) / len(base_entropies)
            delta_pct = (ge - be) / be * 100 if be != 0 else 0
            print(f"  {preset:<12s}: {total_interventions:>4} interventions, "
                  f"entropy {be:.3f}→{ge:.3f} ({delta_pct:+.1f}%)")
        else:
            print(f"  {preset:<12s}: {total_interventions:>4} interventions")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
