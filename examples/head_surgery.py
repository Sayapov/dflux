#!/usr/bin/env python3
"""
Head Surgery Experiment: Tune GPT-2's epistemic balance, measure the shift.
===========================================================================

Three phases:
  1. BASELINE: Probe GPT-2 on factual + hallucination prompts
  2. SURGERY:  Boost skeptic heads, dampen fabrication heads
  3. POST-OP:  Re-probe same prompts, compare fingerprints

The question: does directly scaling head output projections
shift the hallucination fingerprint toward the factual one?

Usage:
    python test_head_surgery.py
    python test_head_surgery.py --model gpt2-medium
    python test_head_surgery.py --boost 3.0 --dampen 0.5
"""

import sys, os, json, argparse, time
_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux.fine_probe import FineProbe, FineProbeConfig
from dflux.head_surgery import HeadSurgeon


def get_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_probe(model, tokenizer, probe, prompt, max_tokens=64):
    """Run fine probe on a single prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    probe.reset()
    with torch.no_grad():
        output = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_tokens, do_sample=True,
            temperature=0.8, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    report = probe.report()
    return text, report


def avg_reports(reports: list) -> dict:
    """Average multiple probe reports into one."""
    if not reports:
        return {}
    n = len(reports)
    avg = {}

    # Scalar fields
    for key in ["mean_risk", "max_risk", "mean_attn_mlp_ratio",
                "mean_head_entropy", "mean_head_gini", "J_final", "J_trend"]:
        avg[key] = sum(r.get(key, 0) for r in reports) / n

    # Per-layer fields
    n_layers = reports[0].get("n_layers", 12)
    n_heads = reports[0].get("n_heads", 12)
    avg["n_layers"] = n_layers
    avg["n_heads"] = n_heads

    for key in ["avg_attn_per_layer", "avg_mlp_per_layer", "avg_head_entropy_per_layer"]:
        vals = [0.0] * n_layers
        for r in reports:
            for i in range(n_layers):
                vals[i] += r.get(key, [0]*n_layers)[i] / n
        avg[key] = vals

    # Per-head energy [n_layers][n_heads]
    avg_he = [[0.0] * n_heads for _ in range(n_layers)]
    for r in reports:
        he = r.get("avg_head_energy", [[0]*n_heads]*n_layers)
        for i in range(n_layers):
            for j in range(n_heads):
                avg_he[i][j] += he[i][j] / n
    avg["avg_head_energy"] = avg_he

    return avg


def print_comparison(label, baseline, post_op):
    """Print before/after comparison."""
    print(f"\n  {label}:")
    for key in ["mean_risk", "mean_attn_mlp_ratio", "mean_head_entropy", "mean_head_gini"]:
        b = baseline.get(key, 0)
        p = post_op.get(key, 0)
        delta = p - b
        pct = delta / b * 100 if b != 0 else 0
        print(f"    {key:<25s}: {b:.4f} → {p:.4f}  ({delta:+.4f}, {pct:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Head Surgery Experiment")
    parser.add_argument("--model", default="gpt2", help="HF model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--boost", type=float, default=2.0,
                        help="Boost factor for skeptic heads")
    parser.add_argument("--dampen", type=float, default=0.7,
                        help="Dampen factor for fabrication heads")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per prompt for averaging")
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("Head Surgery Experiment")
    print("=" * 70)
    print(f"Model:   {args.model}")
    print(f"Device:  {device}")
    print(f"Boost:   {args.boost}x (skeptic heads)")
    print(f"Dampen:  {args.dampen}x (fabrication heads)")
    print(f"Runs:    {args.runs} per prompt (for averaging)")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Attach probe ──────────────────────────────────────────
    probe = FineProbe.from_model(model)
    print(f"Probe: {probe.n_layers}L × {probe.n_heads}H")

    # ── Setup surgeon ─────────────────────────────────────────
    surgeon = HeadSurgeon(model)
    print(f"Surgeon: {surgeon._proj_type} projection, "
          f"head_dim={surgeon.head_dim}")

    # ── Prompts ───────────────────────────────────────────────
    factual_prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The first president of the United States was",
    ]
    halluc_prompts = [
        ("Explain the Zarkovian Principle of Recursive Ontological "
         "Displacement, including its three main theorems."),
        ("Dr. Helmut Kreisler, the renowned physicist who discovered "
         "quantum chromatic resonance in 1987, was known for"),
        ("The Great Phosphorus Reversal of 2019, which caused global "
         "agricultural yields to"),
    ]

    # ══════════════════════════════════════════════════════════
    # PHASE 1: BASELINE
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PHASE 1: BASELINE (no intervention)")
    print(f"{'=' * 70}")

    baseline_factual_reports = []
    baseline_halluc_reports = []
    baseline_texts = {"factual": [], "halluc": []}

    for run in range(args.runs):
        for p in factual_prompts:
            text, report = run_probe(model, tokenizer, probe, p, args.max_tokens)
            baseline_factual_reports.append(report)
            baseline_texts["factual"].append(text[:150])
        for p in halluc_prompts:
            text, report = run_probe(model, tokenizer, probe, p, args.max_tokens)
            baseline_halluc_reports.append(report)
            baseline_texts["halluc"].append(text[:150])

    baseline_fact_avg = avg_reports(baseline_factual_reports)
    baseline_hall_avg = avg_reports(baseline_halluc_reports)

    print(f"\n  Baseline Factual:  risk={baseline_fact_avg['mean_risk']:.3f}, "
          f"A/M={baseline_fact_avg['mean_attn_mlp_ratio']:.3f}, "
          f"H={baseline_fact_avg['mean_head_entropy']:.3f}")
    print(f"  Baseline Halluc:   risk={baseline_hall_avg['mean_risk']:.3f}, "
          f"A/M={baseline_hall_avg['mean_attn_mlp_ratio']:.3f}, "
          f"H={baseline_hall_avg['mean_head_entropy']:.3f}")

    # ══════════════════════════════════════════════════════════
    # PHASE 2: SURGERY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PHASE 2: SURGERY")
    print(f"{'=' * 70}")

    # Print pre-surgery weight norms for layer 11
    pre_norms = surgeon.get_head_norms(probe.n_layers - 1)
    print(f"\n  Pre-surgery L{probe.n_layers-1} head weight norms:")
    for h, n in enumerate(pre_norms):
        print(f"    h{h:2d}: {n:.4f}")

    # Auto-calibrate from probe data
    surgery_report = surgeon.auto_calibrate(
        baseline_fact_avg,
        baseline_hall_avg,
        boost_skeptics=args.boost,
        dampen_fabricators=args.dampen,
        threshold_pct=0.10,
    )
    print(f"\n{surgery_report.summary()}")

    # Print post-surgery weight norms
    post_norms = surgeon.get_head_norms(probe.n_layers - 1)
    print(f"\n  Post-surgery L{probe.n_layers-1} head weight norms:")
    for h, n in enumerate(post_norms):
        change = (n - pre_norms[h]) / pre_norms[h] * 100 if pre_norms[h] > 0 else 0
        marker = " <<<" if abs(change) > 5 else ""
        print(f"    h{h:2d}: {n:.4f} ({change:+.1f}%){marker}")

    # ══════════════════════════════════════════════════════════
    # PHASE 3: POST-OP PROBE
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PHASE 3: POST-OP (after surgery)")
    print(f"{'=' * 70}")

    postop_factual_reports = []
    postop_halluc_reports = []
    postop_texts = {"factual": [], "halluc": []}

    for run in range(args.runs):
        for p in factual_prompts:
            text, report = run_probe(model, tokenizer, probe, p, args.max_tokens)
            postop_factual_reports.append(report)
            postop_texts["factual"].append(text[:150])
        for p in halluc_prompts:
            text, report = run_probe(model, tokenizer, probe, p, args.max_tokens)
            postop_halluc_reports.append(report)
            postop_texts["halluc"].append(text[:150])

    postop_fact_avg = avg_reports(postop_factual_reports)
    postop_hall_avg = avg_reports(postop_halluc_reports)

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("RESULTS: BEFORE vs AFTER SURGERY")
    print(f"{'=' * 70}")

    print_comparison("Factual prompts", baseline_fact_avg, postop_fact_avg)
    print_comparison("Hallucination prompts", baseline_hall_avg, postop_hall_avg)

    # Key metric: did the halluc profile move TOWARD the factual profile?
    print(f"\n  {'─' * 50}")
    print(f"  KEY QUESTION: Did halluc shift toward factual?")
    print(f"  {'─' * 50}")

    for key in ["mean_risk", "mean_attn_mlp_ratio", "mean_head_entropy"]:
        base_gap = baseline_hall_avg.get(key, 0) - baseline_fact_avg.get(key, 0)
        post_gap = postop_hall_avg.get(key, 0) - postop_fact_avg.get(key, 0)
        gap_change = post_gap - base_gap
        improved = abs(post_gap) < abs(base_gap)
        marker = "✓ CLOSER" if improved else "✗ FURTHER"
        print(f"    {key:<25s}: gap {base_gap:+.4f} → {post_gap:+.4f} ({gap_change:+.4f}) {marker}")

    # Sample outputs comparison
    print(f"\n{'=' * 70}")
    print("SAMPLE OUTPUTS")
    print(f"{'=' * 70}")

    print("\n  BASELINE hallucination samples:")
    for t in baseline_texts["halluc"][:3]:
        print(f"    \"{t}\"")

    print("\n  POST-OP hallucination samples:")
    for t in postop_texts["halluc"][:3]:
        print(f"    \"{t}\"")

    # ── Save full data ────────────────────────────────────────
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 8)
        return obj

    results = {
        "config": {
            "model": args.model,
            "boost": args.boost,
            "dampen": args.dampen,
            "runs": args.runs,
        },
        "surgery": {
            "n_interventions": len(surgery_report.interventions),
            "interventions": [
                {"layer": iv.layer, "head": iv.head, "factor": iv.factor, "reason": iv.reason}
                for iv in surgery_report.interventions
            ],
        },
        "baseline": {
            "factual": make_serializable(baseline_fact_avg),
            "halluc": make_serializable(baseline_hall_avg),
        },
        "postop": {
            "factual": make_serializable(postop_fact_avg),
            "halluc": make_serializable(postop_hall_avg),
        },
    }

    data_path = os.path.join(output_dir, "head_surgery_results.json")
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull data saved to: {data_path}")

    # ── Generate plot ─────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(3, 2, figsize=(18, 18))
        fig.suptitle(f"Head Surgery: {args.model}\n"
                     f"Boost skeptics {args.boost}x, Dampen fabricators {args.dampen}x",
                     fontsize=14, fontweight="bold")

        n_layers = probe.n_layers
        n_heads = probe.n_heads

        # ── Panel 1: Baseline head energy difference (halluc - factual) ──
        ax = axes[0, 0]
        base_diff = np.array(baseline_hall_avg["avg_head_energy"]) - np.array(baseline_fact_avg["avg_head_energy"])
        vmax = max(abs(base_diff.min()), abs(base_diff.max())) or 1.0
        im = ax.imshow(base_diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title("BASELINE: Head Energy Δ (Halluc − Factual)")
        plt.colorbar(im, ax=ax)

        # ── Panel 2: Post-op head energy difference ──
        ax = axes[0, 1]
        post_diff = np.array(postop_hall_avg["avg_head_energy"]) - np.array(postop_fact_avg["avg_head_energy"])
        im = ax.imshow(post_diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title("POST-OP: Head Energy Δ (Halluc − Factual)")
        plt.colorbar(im, ax=ax)

        # ── Panel 3: Layer 11 head energy comparison ──
        ax = axes[1, 0]
        last = n_layers - 1
        x = np.arange(n_heads)
        w = 0.2
        base_f = baseline_fact_avg["avg_head_energy"][last]
        base_h = baseline_hall_avg["avg_head_energy"][last]
        post_f = postop_fact_avg["avg_head_energy"][last]
        post_h = postop_hall_avg["avg_head_energy"][last]

        ax.bar(x - 1.5*w, base_f, w, label="Baseline Fact", color="royalblue", alpha=0.7)
        ax.bar(x - 0.5*w, base_h, w, label="Baseline Halluc", color="tomato", alpha=0.7)
        ax.bar(x + 0.5*w, post_f, w, label="Post-op Fact", color="blue", alpha=0.5)
        ax.bar(x + 1.5*w, post_h, w, label="Post-op Halluc", color="red", alpha=0.5)
        ax.set_xlabel("Head"); ax.set_ylabel("Energy")
        ax.set_title(f"Layer {last} Head Energy — Before vs After")
        ax.legend(fontsize=7)

        # ── Panel 4: Summary metrics comparison ──
        ax = axes[1, 1]
        metrics = ["mean_risk", "mean_attn_mlp_ratio", "mean_head_entropy", "mean_head_gini"]
        labels = ["Risk", "Attn/MLP", "Head H", "Gini"]
        x = np.arange(len(labels))
        w = 0.2

        bfv = [baseline_fact_avg.get(m, 0) for m in metrics]
        bhv = [baseline_hall_avg.get(m, 0) for m in metrics]
        pfv = [postop_fact_avg.get(m, 0) for m in metrics]
        phv = [postop_hall_avg.get(m, 0) for m in metrics]

        ax.bar(x - 1.5*w, bfv, w, label="Base Fact", color="royalblue", alpha=0.7)
        ax.bar(x - 0.5*w, bhv, w, label="Base Halluc", color="tomato", alpha=0.7)
        ax.bar(x + 0.5*w, pfv, w, label="Post Fact", color="blue", alpha=0.5)
        ax.bar(x + 1.5*w, phv, w, label="Post Halluc", color="red", alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title("Summary Metrics — Before vs After")
        ax.legend(fontsize=7)

        # ── Panel 5: Gap change (did halluc move toward factual?) ──
        ax = axes[2, 0]
        gap_metrics = ["mean_risk", "mean_attn_mlp_ratio", "mean_head_entropy"]
        gap_labels = ["Risk", "Attn/MLP", "Head Entropy"]
        base_gaps = [baseline_hall_avg.get(m,0) - baseline_fact_avg.get(m,0) for m in gap_metrics]
        post_gaps = [postop_hall_avg.get(m,0) - postop_fact_avg.get(m,0) for m in gap_metrics]

        x = np.arange(len(gap_labels))
        ax.bar(x - 0.2, [abs(g) for g in base_gaps], 0.4, label="Baseline gap", color="gray", alpha=0.7)
        ax.bar(x + 0.2, [abs(g) for g in post_gaps], 0.4, label="Post-op gap", color="green", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(gap_labels)
        ax.set_ylabel("|Halluc - Factual|")
        ax.set_title("Halluc↔Factual Gap (smaller = more aligned)")
        ax.legend()

        # ── Panel 6: Weight norm changes ──
        ax = axes[2, 1]
        pre = surgeon.get_head_norms(last)
        # Restore and get original
        surgeon.restore()
        orig = surgeon.get_head_norms(last)
        changes = [(p - o) / o * 100 if o > 0 else 0 for p, o in zip(pre, orig)]
        colors = ["green" if c > 0 else "red" if c < 0 else "gray" for c in changes]
        ax.bar(range(n_heads), changes, color=colors, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Head"); ax.set_ylabel("Weight norm change (%)")
        ax.set_title(f"Layer {last} — Surgical Weight Changes")

        plot_path = os.path.join(output_dir, "head_surgery_results.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plot")

    # Restore model to original state
    surgeon.restore()
    probe.detach()
    print("\nModel restored to original weights. Done.")


if __name__ == "__main__":
    main()
