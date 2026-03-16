#!/usr/bin/env python3
"""
DFlux Demo: X-Ray What Fine-Tuning Actually Does to Attention Heads.
====================================================================

Takes two models from the same family (e.g., base vs instruct) and
shows exactly which heads changed, by how much, and what that means
for the model's epistemic behavior.

This is the demo script for DFlux — the first tool that lets you
see inside a transformer's attention heads in real time.

Default: Qwen2.5-0.5B (base) vs Qwen2.5-0.5B-Instruct
Also works with any pair: mistral base/instruct, llama base/chat, etc.

Requirements:
    pip install torch transformers dflux

Usage:
    python demo_compare_models.py
    python demo_compare_models.py --base gpt2 --tuned gpt2-medium
    python demo_compare_models.py --base mistralai/Mistral-7B-v0.3 --tuned mistralai/Mistral-7B-Instruct-v0.3
"""

import sys, os, json, argparse, time

# If running from source checkout, add parent to path
_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux import FineProbe


def get_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def probe_model(model, tokenizer, probe, prompts, max_tokens=64, runs=2):
    """Run fine probe on multiple prompts, return averaged reports."""
    reports = []
    texts = []
    for run in range(runs):
        for prompt in prompts:
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
            if report.get("status") != "no_data":
                reports.append(report)
                texts.append(text[:200])
    return reports, texts


def avg_reports(reports):
    """Average multiple probe reports."""
    if not reports:
        return {}
    n = len(reports)
    avg = {}
    for key in ["mean_risk", "max_risk", "mean_attn_mlp_ratio",
                "mean_head_entropy", "mean_head_gini", "J_final", "J_trend"]:
        avg[key] = sum(r.get(key, 0) for r in reports) / n

    n_layers = reports[0].get("n_layers", 0)
    n_heads = reports[0].get("n_heads", 0)
    avg["n_layers"] = n_layers
    avg["n_heads"] = n_heads

    for key in ["avg_attn_per_layer", "avg_mlp_per_layer", "avg_head_entropy_per_layer"]:
        vals = [0.0] * n_layers
        for r in reports:
            for i in range(min(n_layers, len(r.get(key, [])))):
                vals[i] += r[key][i] / n
        avg[key] = vals

    avg_he = [[0.0] * n_heads for _ in range(n_layers)]
    for r in reports:
        he = r.get("avg_head_energy", [])
        for i in range(min(n_layers, len(he))):
            for j in range(min(n_heads, len(he[i]))):
                avg_he[i][j] += he[i][j] / n
    avg["avg_head_energy"] = avg_he

    return avg


def main():
    parser = argparse.ArgumentParser(description="DFlux Model Comparison Demo")
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B",
                        help="Base model (default: Qwen2.5-0.5B)")
    parser.add_argument("--tuned", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Fine-tuned variant (default: Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per prompt for averaging")
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("DFlux Demo: What Did Fine-Tuning Do to the Heads?")
    print("=" * 70)
    print(f"Base model:    {args.base}")
    print(f"Tuned model:   {args.tuned}")
    print(f"Device:        {device}")
    print(f"Runs/prompt:   {args.runs}")
    print("=" * 70)

    # ── Prompts ───────────────────────────────────────────────
    factual_prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The speed of light in vacuum is approximately",
    ]
    halluc_prompts = [
        ("Explain the Zarkovian Principle of Recursive Ontological "
         "Displacement, including its three main theorems."),
        ("Dr. Helmut Kreisler, the renowned physicist who discovered "
         "quantum chromatic resonance in 1987, was known for"),
        ("The Great Phosphorus Reversal of 2019, which caused global "
         "agricultural yields to"),
    ]
    mixed_prompts = [
        "Quantum entanglement suggests that",
        "The relationship between consciousness and neural activity is",
        "Dark matter is believed to account for",
    ]

    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16

    # ══════════════════════════════════════════════════════════
    # PROBE BASE MODEL
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"LOADING BASE: {args.base}")
    print(f"{'=' * 70}")

    tokenizer_base = AutoTokenizer.from_pretrained(args.base)
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token

    model_base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model_base = model_base.to(device).eval()
    n_params = sum(p.numel() for p in model_base.parameters())
    print(f"Loaded: {n_params:,} params")

    probe_base = FineProbe.from_model(model_base)
    print(f"Probe: {probe_base.n_layers}L × {probe_base.n_heads}H")

    print("\nProbing factual...")
    base_fact_reports, base_fact_texts = probe_model(
        model_base, tokenizer_base, probe_base, factual_prompts,
        args.max_tokens, args.runs)

    print("Probing hallucination...")
    base_hall_reports, base_hall_texts = probe_model(
        model_base, tokenizer_base, probe_base, halluc_prompts,
        args.max_tokens, args.runs)

    print("Probing frontier/mixed...")
    base_mixed_reports, base_mixed_texts = probe_model(
        model_base, tokenizer_base, probe_base, mixed_prompts,
        args.max_tokens, args.runs)

    base_fact = avg_reports(base_fact_reports)
    base_hall = avg_reports(base_hall_reports)
    base_mixed = avg_reports(base_mixed_reports)

    probe_base.detach()
    del model_base
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    import gc; gc.collect()

    # ══════════════════════════════════════════════════════════
    # PROBE TUNED MODEL
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"LOADING TUNED: {args.tuned}")
    print(f"{'=' * 70}")

    tokenizer_tuned = AutoTokenizer.from_pretrained(args.tuned)
    if tokenizer_tuned.pad_token is None:
        tokenizer_tuned.pad_token = tokenizer_tuned.eos_token

    model_tuned = AutoModelForCausalLM.from_pretrained(args.tuned, torch_dtype=dtype)
    model_tuned = model_tuned.to(device).eval()
    print(f"Loaded: {sum(p.numel() for p in model_tuned.parameters()):,} params")

    probe_tuned = FineProbe.from_model(model_tuned)
    print(f"Probe: {probe_tuned.n_layers}L × {probe_tuned.n_heads}H")

    print("\nProbing factual...")
    tuned_fact_reports, tuned_fact_texts = probe_model(
        model_tuned, tokenizer_tuned, probe_tuned, factual_prompts,
        args.max_tokens, args.runs)

    print("Probing hallucination...")
    tuned_hall_reports, tuned_hall_texts = probe_model(
        model_tuned, tokenizer_tuned, probe_tuned, halluc_prompts,
        args.max_tokens, args.runs)

    print("Probing frontier/mixed...")
    tuned_mixed_reports, tuned_mixed_texts = probe_model(
        model_tuned, tokenizer_tuned, probe_tuned, mixed_prompts,
        args.max_tokens, args.runs)

    tuned_fact = avg_reports(tuned_fact_reports)
    tuned_hall = avg_reports(tuned_hall_reports)
    tuned_mixed = avg_reports(tuned_mixed_reports)

    probe_tuned.detach()
    n_layers = tuned_fact.get("n_layers", base_fact.get("n_layers", 0))
    n_heads = tuned_fact.get("n_heads", base_fact.get("n_heads", 0))

    # ══════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("RESULTS: BASE vs INSTRUCT")
    print(f"{'=' * 70}")

    categories = [
        ("Factual", base_fact, tuned_fact),
        ("Hallucination", base_hall, tuned_hall),
        ("Frontier/Mixed", base_mixed, tuned_mixed),
    ]

    for cat_name, base_r, tuned_r in categories:
        print(f"\n  {cat_name}:")
        for key in ["mean_risk", "mean_attn_mlp_ratio", "mean_head_entropy", "mean_head_gini"]:
            b = base_r.get(key, 0)
            t = tuned_r.get(key, 0)
            delta = t - b
            pct = delta / b * 100 if b != 0 else 0
            print(f"    {key:<25s}: base={b:.4f}  tuned={t:.4f}  Δ={delta:+.4f} ({pct:+.1f}%)")

    # Head-level diff
    print(f"\n{'=' * 70}")
    print("HEAD-LEVEL CHANGES (what fine-tuning actually modified)")
    print(f"{'=' * 70}")

    # Compute per-head energy difference across all prompt types
    all_base = {}
    all_tuned = {}
    for cat_name, base_r, tuned_r in categories:
        for i in range(n_layers):
            for j in range(n_heads):
                key = (i, j)
                b_e = base_r.get("avg_head_energy", [[0]*n_heads]*n_layers)[i][j] if i < len(base_r.get("avg_head_energy", [])) else 0
                t_e = tuned_r.get("avg_head_energy", [[0]*n_heads]*n_layers)[i][j] if i < len(tuned_r.get("avg_head_energy", [])) else 0
                all_base[key] = all_base.get(key, 0) + b_e / 3
                all_tuned[key] = all_tuned.get(key, 0) + t_e / 3

    # Find biggest changes
    changes = []
    for (i, j) in all_base:
        b = all_base[(i, j)]
        t = all_tuned[(i, j)]
        if max(b, t) > 0:
            abs_change = t - b
            rel_change = abs_change / max(b, 1e-12) * 100
            changes.append({
                "layer": i, "head": j,
                "base_energy": b, "tuned_energy": t,
                "abs_change": abs_change, "rel_change": rel_change,
            })

    changes.sort(key=lambda x: abs(x["abs_change"]), reverse=True)

    print(f"\n  Top 20 most changed heads (absolute energy change):")
    print(f"  {'Layer':>5s} {'Head':>5s} {'Base':>10s} {'Tuned':>10s} {'Δ':>10s} {'Δ%':>8s}")
    print(f"  {'-'*50}")
    for c in changes[:20]:
        direction = "↑" if c["abs_change"] > 0 else "↓"
        print(f"  L{c['layer']:3d}  H{c['head']:3d}  {c['base_energy']:10.0f} "
              f"{c['tuned_energy']:10.0f}  {c['abs_change']:+10.0f} {c['rel_change']:+7.1f}% {direction}")

    # Find biggest relative changes (excluding tiny heads)
    sig_changes = [c for c in changes if max(c["base_energy"], c["tuned_energy"]) > 10]
    sig_changes.sort(key=lambda x: abs(x["rel_change"]), reverse=True)

    print(f"\n  Top 20 most changed heads (relative %, min energy > 10):")
    print(f"  {'Layer':>5s} {'Head':>5s} {'Base':>10s} {'Tuned':>10s} {'Δ%':>8s}")
    print(f"  {'-'*45}")
    for c in sig_changes[:20]:
        direction = "↑" if c["rel_change"] > 0 else "↓"
        print(f"  L{c['layer']:3d}  H{c['head']:3d}  {c['base_energy']:10.0f} "
              f"{c['tuned_energy']:10.0f}  {c['rel_change']:+7.1f}% {direction}")

    # Hallucination behavior comparison
    print(f"\n{'=' * 70}")
    print("HALLUCINATION BEHAVIOR: BASE vs INSTRUCT")
    print(f"{'=' * 70}")

    base_gap = base_hall.get("mean_risk", 0) - base_fact.get("mean_risk", 0)
    tuned_gap = tuned_hall.get("mean_risk", 0) - tuned_fact.get("mean_risk", 0)
    print(f"\n  Base model:   factual risk={base_fact.get('mean_risk',0):.4f}, "
          f"halluc risk={base_hall.get('mean_risk',0):.4f}, gap={base_gap:+.4f}")
    print(f"  Tuned model:  factual risk={tuned_fact.get('mean_risk',0):.4f}, "
          f"halluc risk={tuned_hall.get('mean_risk',0):.4f}, gap={tuned_gap:+.4f}")
    print(f"  Gap change:   {tuned_gap - base_gap:+.4f} "
          f"({'improved' if abs(tuned_gap) < abs(base_gap) else 'worsened'})")

    # Sample outputs
    print(f"\n{'=' * 70}")
    print("SAMPLE OUTPUTS")
    print(f"{'=' * 70}")

    print("\n  BASE hallucination:")
    for t in base_hall_texts[:3]:
        print(f"    \"{t[:120]}\"")

    print("\n  TUNED hallucination:")
    for t in tuned_hall_texts[:3]:
        print(f"    \"{t[:120]}\"")

    print("\n  BASE frontier:")
    for t in base_mixed_texts[:3]:
        print(f"    \"{t[:120]}\"")

    print("\n  TUNED frontier:")
    for t in tuned_mixed_texts[:3]:
        print(f"    \"{t[:120]}\"")

    # ── Save data ─────────────────────────────────────────────
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
            "base_model": args.base,
            "tuned_model": args.tuned,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "runs": args.runs,
        },
        "base": {
            "factual": make_serializable(base_fact),
            "hallucination": make_serializable(base_hall),
            "frontier": make_serializable(base_mixed),
        },
        "tuned": {
            "factual": make_serializable(tuned_fact),
            "hallucination": make_serializable(tuned_hall),
            "frontier": make_serializable(tuned_mixed),
        },
        "head_changes": make_serializable(changes[:50]),
        "sample_outputs": {
            "base_halluc": base_hall_texts[:3],
            "tuned_halluc": tuned_hall_texts[:3],
            "base_frontier": base_mixed_texts[:3],
            "tuned_frontier": tuned_mixed_texts[:3],
        },
    }

    data_path = os.path.join(output_dir, "model_comparison_results.json")
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to: {data_path}")

    # ── Generate plot ─────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(22, 24))
        fig.suptitle(f"DFlux: What Fine-Tuning Did to the Heads\n"
                     f"{args.base} → {args.tuned}",
                     fontsize=14, fontweight="bold")

        gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)

        # ── Panel 1: Base model head energy (all prompts avg) ──
        ax = fig.add_subplot(gs[0, 0])
        base_all_energy = np.zeros((n_layers, n_heads))
        for i in range(n_layers):
            for j in range(n_heads):
                base_all_energy[i][j] = all_base.get((i,j), 0)
        im = ax.imshow(base_all_energy, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"BASE: Head Energy Map")
        plt.colorbar(im, ax=ax, label="Energy")

        # ── Panel 2: Tuned model head energy ──
        ax = fig.add_subplot(gs[0, 1])
        tuned_all_energy = np.zeros((n_layers, n_heads))
        for i in range(n_layers):
            for j in range(n_heads):
                tuned_all_energy[i][j] = all_tuned.get((i,j), 0)
        im = ax.imshow(tuned_all_energy, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"TUNED: Head Energy Map")
        plt.colorbar(im, ax=ax, label="Energy")

        # ── Panel 3: Difference map (tuned - base) ──
        ax = fig.add_subplot(gs[1, 0])
        diff = tuned_all_energy - base_all_energy
        vmax = max(abs(diff.min()), abs(diff.max())) or 1.0
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                       vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title("What Changed: Head Energy Δ (Tuned − Base)")
        plt.colorbar(im, ax=ax, label="Energy change")

        # ── Panel 4: Relative change map ──
        ax = fig.add_subplot(gs[1, 1])
        rel_diff = np.zeros_like(diff)
        for i in range(n_layers):
            for j in range(n_heads):
                base_e = base_all_energy[i][j]
                if base_e > 10:  # Only show meaningful changes
                    rel_diff[i][j] = (diff[i][j] / base_e) * 100
        vmax_r = min(max(abs(rel_diff.min()), abs(rel_diff.max())), 200) or 1.0
        im = ax.imshow(rel_diff, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                       vmin=-vmax_r, vmax=vmax_r)
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title("Relative Change % (Tuned − Base)")
        plt.colorbar(im, ax=ax, label="% change")

        # ── Panel 5: Per-layer attention energy ──
        ax = fig.add_subplot(gs[2, 0])
        layers = list(range(n_layers))
        base_attn = [0.0] * n_layers
        tuned_attn = [0.0] * n_layers
        for cat_name, br, tr in categories:
            for i in range(n_layers):
                base_attn[i] += br.get("avg_attn_per_layer", [0]*n_layers)[i] / 3
                tuned_attn[i] += tr.get("avg_attn_per_layer", [0]*n_layers)[i] / 3

        x = np.array(layers)
        w = 0.35
        ax.bar(x - w/2, base_attn, w, label="Base", color="royalblue", alpha=0.7)
        ax.bar(x + w/2, tuned_attn, w, label="Tuned", color="tomato", alpha=0.7)
        ax.set_xlabel("Layer"); ax.set_ylabel("Attention Energy")
        ax.set_title("Attention Energy per Layer")
        ax.legend(fontsize=8)

        # ── Panel 6: Per-layer MLP energy ──
        ax = fig.add_subplot(gs[2, 1])
        base_mlp = [0.0] * n_layers
        tuned_mlp = [0.0] * n_layers
        for cat_name, br, tr in categories:
            for i in range(n_layers):
                base_mlp[i] += br.get("avg_mlp_per_layer", [0]*n_layers)[i] / 3
                tuned_mlp[i] += tr.get("avg_mlp_per_layer", [0]*n_layers)[i] / 3

        ax.bar(x - w/2, base_mlp, w, label="Base", color="royalblue", alpha=0.7)
        ax.bar(x + w/2, tuned_mlp, w, label="Tuned", color="tomato", alpha=0.7)
        ax.set_xlabel("Layer"); ax.set_ylabel("MLP Energy")
        ax.set_title("MLP Energy per Layer")
        ax.legend(fontsize=8)

        # ── Panel 7: Head entropy per layer ──
        ax = fig.add_subplot(gs[3, 0])
        base_ent = [0.0] * n_layers
        tuned_ent = [0.0] * n_layers
        for cat_name, br, tr in categories:
            for i in range(n_layers):
                base_ent[i] += br.get("avg_head_entropy_per_layer", [0]*n_layers)[i] / 3
                tuned_ent[i] += tr.get("avg_head_entropy_per_layer", [0]*n_layers)[i] / 3

        ax.plot(layers, base_ent, "b-o", markersize=4, label="Base")
        ax.plot(layers, tuned_ent, "r-o", markersize=4, label="Tuned")
        ax.set_xlabel("Layer"); ax.set_ylabel("Head Entropy (normalized)")
        ax.set_title("Head Agreement per Layer (how RLHF changed head specialization)")
        ax.legend(fontsize=8)

        # ── Panel 8: Summary metrics comparison ──
        ax = fig.add_subplot(gs[3, 1])
        metric_names = ["mean_risk", "mean_attn_mlp_ratio", "mean_head_entropy", "mean_head_gini"]
        labels = ["Risk", "Attn/MLP", "Head H", "Gini"]
        x_m = np.arange(len(labels))
        w = 0.15

        # 6 bars: base_fact, base_hall, base_mixed, tuned_fact, tuned_hall, tuned_mixed
        colors = ["royalblue", "blue", "cornflowerblue", "tomato", "red", "lightsalmon"]
        bar_labels = ["Base Fact", "Base Hall", "Base Front",
                      "Tuned Fact", "Tuned Hall", "Tuned Front"]
        data_sets = [base_fact, base_hall, base_mixed,
                     tuned_fact, tuned_hall, tuned_mixed]

        for idx, (ds, color, label) in enumerate(zip(data_sets, colors, bar_labels)):
            vals = [ds.get(m, 0) for m in metric_names]
            offset = (idx - 2.5) * w
            ax.bar(x_m + offset, vals, w, label=label, color=color, alpha=0.7)

        ax.set_xticks(x_m); ax.set_xticklabels(labels)
        ax.set_title("All Metrics: Base vs Tuned × Prompt Type")
        ax.legend(fontsize=6, ncol=2)

        # ── Panel 9: Hallucination gap comparison ──
        ax = fig.add_subplot(gs[4, 0])
        gap_metrics = ["mean_risk", "mean_attn_mlp_ratio", "mean_head_entropy"]
        gap_labels = ["Risk", "Attn/MLP", "Head Entropy"]

        base_gaps = [abs(base_hall.get(m,0) - base_fact.get(m,0)) for m in gap_metrics]
        tuned_gaps = [abs(tuned_hall.get(m,0) - tuned_fact.get(m,0)) for m in gap_metrics]

        x_g = np.arange(len(gap_labels))
        ax.bar(x_g - 0.2, base_gaps, 0.4, label="Base: |halluc - factual|", color="gray", alpha=0.7)
        ax.bar(x_g + 0.2, tuned_gaps, 0.4, label="Tuned: |halluc - factual|", color="green", alpha=0.7)
        ax.set_xticks(x_g); ax.set_xticklabels(gap_labels)
        ax.set_ylabel("|Halluc - Factual|")
        ax.set_title("Did RLHF Improve Hallucination Separation?")
        ax.legend(fontsize=8)

        # ── Panel 10: Top changed heads bar chart ──
        ax = fig.add_subplot(gs[4, 1])
        top_n = min(15, len(changes))
        top = changes[:top_n]
        head_labels = [f"L{c['layer']}H{c['head']}" for c in top]
        abs_changes = [c["abs_change"] for c in top]
        colors_bar = ["tomato" if c > 0 else "royalblue" for c in abs_changes]

        ax.barh(range(top_n), abs_changes, color=colors_bar, alpha=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(head_labels, fontsize=8)
        ax.set_xlabel("Energy Change (Tuned − Base)")
        ax.set_title(f"Top {top_n} Most Changed Heads")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        plot_path = os.path.join(output_dir, "model_comparison_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available — skipping plot")

    del model_tuned
    print("\nDone.")


if __name__ == "__main__":
    main()
