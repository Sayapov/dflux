#!/usr/bin/env python3
"""
Causal Primitives: What do GPT-2's heads ACTUALLY do?
=====================================================

Instead of guessing which heads are "skeptics" or "arbitrators,"
we measure each head's causal contribution information-theoretically.

CP = I(head_energy; output_metric) / log2(n_bins)

High CP: this head reliably and uniquely drives output behavior.
Low CP: this head is noise or redundant.

Also computes:
  - Emergent hierarchy: is causation top-heavy, bottom-heavy, or distributed?
  - Cross-head interactions: which heads cause changes in which other heads?
  - CP comparison between factual vs hallucination prompts

Usage:
    python causal_primitives_gpt2.py
    python causal_primitives_gpt2.py --model gpt2-medium
"""

import sys, os, json, argparse

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux import FineProbe
from dflux.causal_primitives import CausalPrimitives, CPConfig, compute_cross_head_cp


def probe_and_collect(model, tokenizer, probe, prompts, max_tokens=64, runs=2):
    """Run probe and return token-level diagnostics for CP analysis."""
    all_diagnostics = []
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
            # Collect token diagnostics
            all_diagnostics.extend(probe.diagnostics)
    return all_diagnostics


def diagnostics_to_cp(diagnostics, n_layers, n_heads, cfg=None):
    """Convert FineProbe diagnostics into CP report."""
    cp = CausalPrimitives(n_layers, n_heads, cfg or CPConfig(n_bins=16))
    for d in diagnostics:
        he = d.head_energies
        le = [sum(heads) for heads in he] if he else []
        cp.observe_token(he, le, d.J)
    return cp.compute()


def diagnostics_to_cross_head(diagnostics, n_layers, n_heads):
    """Compute cross-head interactions from diagnostics."""
    # Reshape to [n_layers][n_heads][n_tokens]
    head_energies = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    for d in diagnostics:
        for i in range(min(n_layers, len(d.head_energies))):
            for j in range(min(n_heads, len(d.head_energies[i]))):
                head_energies[i][j].append(d.head_energies[i][j])
    return compute_cross_head_cp(head_energies, n_layers, n_heads)


def main():
    parser = argparse.ArgumentParser(description="Causal Primitives on GPT-2")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--bins", type=int, default=16)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("Causal Primitives: What Do the Heads Actually Do?")
    print("=" * 70)
    print(f"Model:     {args.model}")
    print(f"Device:    {device}")
    print(f"Bins:      {args.bins}")
    print(f"Runs:      {args.runs}")
    print("=" * 70)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model = model.to(device).eval()

    probe = FineProbe.from_model(model)
    n_layers = probe.n_layers
    n_heads = probe.n_heads
    print(f"Architecture: {n_layers}L × {n_heads}H = {n_layers * n_heads} heads")

    cfg = CPConfig(n_bins=args.bins)

    # ── Factual prompts ──
    factual_prompts = [
        "The capital of France is",
        "Water boils at a temperature of",
        "The speed of light in vacuum is approximately",
        "The chemical formula for water is",
        "The Earth orbits the Sun in approximately",
    ]

    # ── Hallucination prompts ──
    halluc_prompts = [
        ("Explain the Zarkovian Principle of Recursive Ontological "
         "Displacement, including its three main theorems."),
        ("Dr. Helmut Kreisler, the renowned physicist who discovered "
         "quantum chromatic resonance in 1987, was known for"),
        ("The Great Phosphorus Reversal of 2019, which caused global "
         "agricultural yields to"),
        "The Hendricks-Maslow Equation for cognitive load states that",
        "The city of Pralnikov, founded in 1823, is famous for",
    ]

    # ── Mixed/frontier prompts ──
    mixed_prompts = [
        "Quantum entanglement suggests that",
        "The relationship between consciousness and neural activity is",
        "Dark matter is believed to account for",
    ]

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Probe all prompt categories
    # ══════════════════════════════════════════════════════════

    print("\n--- Probing factual prompts ---")
    fact_diags = probe_and_collect(
        model, tokenizer, probe, factual_prompts, args.max_tokens, args.runs)
    print(f"  Collected {len(fact_diags)} token diagnostics")

    print("--- Probing hallucination prompts ---")
    hall_diags = probe_and_collect(
        model, tokenizer, probe, halluc_prompts, args.max_tokens, args.runs)
    print(f"  Collected {len(hall_diags)} token diagnostics")

    print("--- Probing frontier prompts ---")
    mixed_diags = probe_and_collect(
        model, tokenizer, probe, mixed_prompts, args.max_tokens, args.runs)
    print(f"  Collected {len(mixed_diags)} token diagnostics")

    # Combine all diagnostics for overall CP
    all_diags = fact_diags + hall_diags + mixed_diags
    print(f"\nTotal tokens: {len(all_diags)}")

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Compute CP at all scales
    # ══════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print("OVERALL CAUSAL PRIMITIVES")
    print(f"{'=' * 70}")

    overall_cp = diagnostics_to_cp(all_diags, n_layers, n_heads, cfg)
    print(f"\n  Hierarchy:  {overall_cp['hierarchy']}")
    print(f"  Emergence:  {overall_cp['emergence']:.4f}")
    print(f"  S_path:     {overall_cp['S_path_norm']:.4f} (spread across layers)")
    print(f"  S_row_bar:  {overall_cp['S_row_bar']:.4f} (differentiation within layers)")

    print(f"\n  Layer CP (where does causation live?):")
    for i in range(n_layers):
        bar = "█" * int(overall_cp['layer_cp'][i] * 100)
        print(f"    L{i:2d}: CP={overall_cp['layer_cp'][i]:.4f}  "
              f"det={overall_cp['layer_det'][i]:.4f}  "
              f"spec={overall_cp['layer_spec'][i]:.4f}  {bar}")

    print(f"\n  Top 15 heads by causal contribution:")
    print(f"  {'Layer':>5s} {'Head':>5s} {'CP':>8s} {'Determ':>8s} {'Specif':>8s} {'MI':>8s} {'Energy':>8s}")
    print(f"  {'-'*50}")
    for h in overall_cp['top_heads'][:15]:
        print(f"  L{h['layer']:3d}  H{h['head']:3d}  {h['cp']:.4f}   "
              f"{h['determinism']:.4f}   {h['specificity']:.4f}   "
              f"{h['mutual_info']:.4f}  {h['mean_energy']:.2f}")

    print(f"\n  Bottom 5 heads (lowest causal contribution — noise or redundant):")
    for h in overall_cp['bottom_heads'][:5]:
        print(f"  L{h['layer']:3d}  H{h['head']:3d}  CP={h['cp']:.4f}  "
              f"det={h['determinism']:.4f}  spec={h['specificity']:.4f}  "
              f"energy={h['mean_energy']:.2f}")

    # ══════════════════════════════════════════════════════════
    # PHASE 3: Compare CP across prompt types
    # ══════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print("CP BY PROMPT TYPE — Does causal structure shift?")
    print(f"{'=' * 70}")

    fact_cp = diagnostics_to_cp(fact_diags, n_layers, n_heads, cfg)
    hall_cp = diagnostics_to_cp(hall_diags, n_layers, n_heads, cfg)
    mixed_cp = diagnostics_to_cp(mixed_diags, n_layers, n_heads, cfg)

    for name, cp_report in [("Factual", fact_cp), ("Hallucination", hall_cp), ("Frontier", mixed_cp)]:
        if cp_report.get("status") != "ok":
            print(f"\n  {name}: insufficient data")
            continue
        print(f"\n  {name}:")
        print(f"    Hierarchy:  {cp_report['hierarchy']}")
        print(f"    Emergence:  {cp_report['emergence']:.4f}")
        print(f"    Top 5 heads: ", end="")
        for h in cp_report['top_heads'][:5]:
            print(f"L{h['layer']}H{h['head']}({h['cp']:.3f}) ", end="")
        print()

    # CP shift: which heads change causal role between factual and hallucination?
    if fact_cp.get("status") == "ok" and hall_cp.get("status") == "ok":
        print(f"\n  CP SHIFT (Halluc - Factual):")
        print(f"  Heads that gain causal influence during hallucination:")
        shifts = []
        for i in range(n_layers):
            for j in range(n_heads):
                f_cp = fact_cp['head_cp'][i][j]
                h_cp = hall_cp['head_cp'][i][j]
                delta = h_cp - f_cp
                shifts.append({"layer": i, "head": j, "fact_cp": f_cp,
                               "hall_cp": h_cp, "delta": delta})

        shifts.sort(key=lambda x: x["delta"], reverse=True)

        print(f"  {'Layer':>5s} {'Head':>5s} {'Fact CP':>8s} {'Hall CP':>8s} {'Δ':>8s}")
        print(f"  {'-'*40}")
        for s in shifts[:10]:
            direction = "↑" if s["delta"] > 0 else "↓"
            print(f"  L{s['layer']:3d}  H{s['head']:3d}  {s['fact_cp']:.4f}   "
                  f"{s['hall_cp']:.4f}  {s['delta']:+.4f} {direction}")

        print(f"\n  Heads that LOSE causal influence during hallucination:")
        for s in shifts[-10:]:
            direction = "↑" if s["delta"] > 0 else "↓"
            print(f"  L{s['layer']:3d}  H{s['head']:3d}  {s['fact_cp']:.4f}   "
                  f"{s['hall_cp']:.4f}  {s['delta']:+.4f} {direction}")

    # ══════════════════════════════════════════════════════════
    # PHASE 4: Cross-head interactions
    # ══════════════════════════════════════════════════════════

    print(f"\n{'=' * 70}")
    print("CROSS-HEAD CAUSAL INTERACTIONS")
    print(f"{'=' * 70}")

    cross = diagnostics_to_cross_head(all_diags, n_layers, n_heads)
    if cross.get("status") == "ok":
        print(f"\n  {cross['n_interactions']} significant interactions found")
        print(f"\n  Top 15 head-to-head causal links:")
        print(f"  {'Cause':>10s} → {'Effect':>10s}  {'MI':>8s}  {'CP':>8s}")
        print(f"  {'-'*45}")
        for ix in cross['top_interactions'][:15]:
            print(f"  L{ix['cause_layer']}H{ix['cause_head']:2d}     → "
                  f"L{ix['effect_layer']}H{ix['effect_head']:2d}      "
                  f"{ix['mutual_info']:.4f}   {ix['cp']:.4f}")
    else:
        print(f"  {cross.get('status', 'unknown')}")

    # ── Save results ──────────────────────────────────────────
    results = {
        "config": {
            "model": args.model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_bins": args.bins,
            "runs": args.runs,
            "total_tokens": len(all_diags),
        },
        "overall": overall_cp,
        "factual": fact_cp,
        "hallucination": hall_cp,
        "frontier": mixed_cp,
        "cross_head": {
            "n_interactions": cross.get("n_interactions", 0),
            "top_interactions": cross.get("top_interactions", []),
        },
    }

    # Compute CP shifts if both are OK
    if fact_cp.get("status") == "ok" and hall_cp.get("status") == "ok":
        results["cp_shifts"] = shifts[:20] + shifts[-20:]

    data_path = os.path.join(output_dir, "causal_primitives_results.json")
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to: {data_path}")

    # ── Generate plot ─────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(22, 20))
        fig.suptitle(f"Causal Primitives: {args.model}\n"
                     f"What do the heads actually do?",
                     fontsize=14, fontweight="bold")

        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

        # Panel 1: Head CP heatmap
        ax = fig.add_subplot(gs[0, 0])
        cp_map = np.array(overall_cp["head_cp"])
        im = ax.imshow(cp_map, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title("Causal Primitives per Head (overall)")
        plt.colorbar(im, ax=ax, label="CP")

        # Panel 2: Head determinism heatmap
        ax = fig.add_subplot(gs[0, 1])
        det_map = np.array(overall_cp["head_det"])
        im = ax.imshow(det_map, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title("Determinism per Head")
        plt.colorbar(im, ax=ax, label="Determinism")

        # Panel 3: Layer CP profile
        ax = fig.add_subplot(gs[1, 0])
        layers = list(range(n_layers))
        ax.bar(layers, overall_cp["layer_cp"], color="steelblue", alpha=0.8)
        ax.set_xlabel("Layer"); ax.set_ylabel("CP")
        ax.set_title(f"Layer Causal Profile — {overall_cp['hierarchy']}")

        # Panel 4: CP comparison across prompt types
        ax = fig.add_subplot(gs[1, 1])
        if fact_cp.get("status") == "ok" and hall_cp.get("status") == "ok":
            x = np.arange(n_layers)
            w = 0.25
            ax.bar(x - w, fact_cp["layer_cp"], w, label="Factual", color="royalblue", alpha=0.7)
            ax.bar(x, hall_cp["layer_cp"], w, label="Hallucination", color="tomato", alpha=0.7)
            if mixed_cp.get("status") == "ok":
                ax.bar(x + w, mixed_cp["layer_cp"], w, label="Frontier", color="green", alpha=0.7)
            ax.set_xlabel("Layer"); ax.set_ylabel("CP")
            ax.set_title("Layer CP by Prompt Type")
            ax.legend(fontsize=8)

        # Panel 5: CP shift heatmap (halluc - factual)
        ax = fig.add_subplot(gs[2, 0])
        if fact_cp.get("status") == "ok" and hall_cp.get("status") == "ok":
            shift_map = np.array(hall_cp["head_cp"]) - np.array(fact_cp["head_cp"])
            vmax = max(abs(shift_map.min()), abs(shift_map.max())) or 0.01
            im = ax.imshow(shift_map, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                           vmin=-vmax, vmax=vmax)
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title("CP Shift: Hallucination − Factual")
            plt.colorbar(im, ax=ax, label="ΔCP")

        # Panel 6: Top heads bar chart
        ax = fig.add_subplot(gs[2, 1])
        top_n = min(20, len(overall_cp["top_heads"]))
        labels = [f"L{h['layer']}H{h['head']}" for h in overall_cp["top_heads"][:top_n]]
        cps = [h["cp"] for h in overall_cp["top_heads"][:top_n]]
        dets = [h["determinism"] for h in overall_cp["top_heads"][:top_n]]

        y = np.arange(top_n)
        ax.barh(y, cps, 0.4, label="CP", color="steelblue", alpha=0.8)
        ax.barh(y + 0.4, dets, 0.4, label="Determinism", color="tomato", alpha=0.6)
        ax.set_yticks(y + 0.2)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Value")
        ax.set_title(f"Top {top_n} Heads by Causal Primitives")
        ax.legend(fontsize=8)
        ax.invert_yaxis()

        # Panel 7: Cross-head interaction network (simplified)
        ax = fig.add_subplot(gs[3, 0])
        if cross.get("status") == "ok" and cross["top_interactions"]:
            top_ix = cross["top_interactions"][:20]
            for ix in top_ix:
                ax.annotate("",
                    xy=(ix["effect_head"], ix["effect_layer"]),
                    xytext=(ix["cause_head"], ix["cause_layer"]),
                    arrowprops=dict(arrowstyle="->",
                                    color="red",
                                    alpha=min(1.0, ix["mutual_info"] * 3),
                                    lw=max(0.5, ix["mutual_info"] * 5)))
            ax.set_xlim(-0.5, n_heads - 0.5)
            ax.set_ylim(n_layers - 0.5, -0.5)
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title("Top 20 Cross-Head Causal Links")
            ax.grid(True, alpha=0.2)

        # Panel 8: Emergence summary
        ax = fig.add_subplot(gs[3, 1])
        ax.axis("off")
        summary = [
            f"Model: {args.model}",
            f"Architecture: {n_layers}L × {n_heads}H = {n_layers * n_heads} heads",
            f"Tokens analyzed: {len(all_diags)}",
            f"",
            f"Emergent Hierarchy: {overall_cp['hierarchy']}",
            f"Emergent Complexity: {overall_cp['emergence']:.4f}",
            f"Path Entropy (S_path): {overall_cp['S_path_norm']:.4f}",
            f"Row Negentropy (S̄_row): {overall_cp['S_row_bar']:.4f}",
            f"",
            f"Top causal head: L{overall_cp['top_heads'][0]['layer']}H{overall_cp['top_heads'][0]['head']} "
            f"(CP={overall_cp['top_heads'][0]['cp']:.4f})",
            f"",
            f"Factual hierarchy: {fact_cp.get('hierarchy', 'N/A')}",
            f"Halluc hierarchy:  {hall_cp.get('hierarchy', 'N/A')}",
        ]
        if fact_cp.get("status") == "ok" and hall_cp.get("status") == "ok":
            summary.append(f"Factual emergence:  {fact_cp['emergence']:.4f}")
            summary.append(f"Halluc emergence:   {hall_cp['emergence']:.4f}")

        ax.text(0.05, 0.95, "\n".join(summary), transform=ax.transAxes,
                fontsize=10, fontfamily="monospace", verticalalignment="top")

        plot_path = os.path.join(output_dir, "causal_primitives_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plot")

    probe.detach()
    print("\nDone.")


if __name__ == "__main__":
    main()
