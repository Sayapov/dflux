#!/usr/bin/env python3
"""
Comprehensive Causal Primitives Analysis
=========================================

Tests CP across multiple prompt categories AND multiple models to answer:
  1. Does causal structure shift by domain (fact, halluc, code, math, creative, reasoning)?
  2. Do different models have different emergent hierarchies?
  3. Does fine-tuning (base vs instruct) change WHERE causation lives?

Usage:
    python cp_comprehensive.py                              # GPT-2 only
    python cp_comprehensive.py --models gpt2                # GPT-2 only
    python cp_comprehensive.py --models qwen                # Qwen only
    python cp_comprehensive.py --models gpt2 qwen           # Both
    python cp_comprehensive.py --models qwen --device mps   # Qwen on Apple Silicon
"""

import sys, os, json, argparse, time, gc

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux import FineProbe
from dflux.causal_primitives import CausalPrimitives, CPConfig, compute_cross_head_cp


# ═══════════════════════════════════════════════════════════
# PROMPT LIBRARY — 8 categories, 4 prompts each
# ═══════════════════════════════════════════════════════════

PROMPT_CATEGORIES = {
    "factual": [
        "The capital of France is",
        "Water boils at a temperature of",
        "The speed of light in vacuum is approximately",
        "The chemical formula for table salt is",
    ],
    "hallucination": [
        ("Explain the Zarkovian Principle of Recursive Ontological "
         "Displacement, including its three main theorems."),
        ("Dr. Helmut Kreisler, the renowned physicist who discovered "
         "quantum chromatic resonance in 1987, was known for"),
        ("The Great Phosphorus Reversal of 2019, which caused global "
         "agricultural yields to"),
        "The Hendricks-Maslow Equation for cognitive load states that",
    ],
    "reasoning": [
        "If all roses are flowers, and some flowers fade quickly, then",
        "A train leaves New York at 60 mph. Another leaves Boston at 80 mph. They will meet when",
        "The logical flaw in the statement 'I always lie' is that",
        "If increasing taxes reduces spending, and reducing spending lowers inflation, then increasing taxes",
    ],
    "math": [
        "The integral of x^2 from 0 to 1 equals",
        "The derivative of sin(x) * cos(x) is",
        "If f(x) = e^(2x), then f'(x) equals",
        "The sum of the first 100 natural numbers is",
    ],
    "code": [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "# Python function to reverse a linked list\ndef reverse_list(head):\n",
        "import numpy as np\n\n# Compute the eigenvalues of a matrix\ndef eigenvalues(matrix):\n",
        "# Binary search implementation\ndef binary_search(arr, target):\n",
    ],
    "creative": [
        "Once upon a time, in a kingdom where shadows could speak,",
        "The astronaut opened the airlock and saw something impossible:",
        "She had been dead for three years when the letter arrived,",
        "The last human on Earth sat alone in a room. There was a knock on the door.",
    ],
    "frontier": [
        "Quantum entanglement suggests that",
        "The relationship between consciousness and neural activity is",
        "Dark matter is believed to account for",
        "The hard problem of consciousness refers to",
    ],
    "instruction": [
        "Write a haiku about the ocean.",
        "Explain photosynthesis to a five-year-old.",
        "List three advantages of renewable energy.",
        "Summarize the plot of Romeo and Juliet in one sentence.",
    ],
}

# Model configurations
MODEL_CONFIGS = {
    "gpt2": {
        "models": [("gpt2", "GPT-2 (124M)")],
    },
    "qwen": {
        "models": [
            ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B Base"),
            ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B Instruct"),
        ],
    },
}


def get_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def probe_category(model, tokenizer, probe, prompts, max_tokens=64, runs=2):
    """Run probe on a category of prompts, return diagnostics."""
    all_diags = []
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
            all_diags.extend(probe.diagnostics)
    return all_diags


def diags_to_cp(diags, n_layers, n_heads, n_bins=16):
    """Convert diagnostics to CP report."""
    cp = CausalPrimitives(n_layers, n_heads, CPConfig(n_bins=n_bins))
    for d in diags:
        he = d.head_energies
        le = [sum(heads) for heads in he] if he else []
        cp.observe_token(he, le, d.J)
    return cp.compute()


def diags_to_cross(diags, n_layers, n_heads):
    """Compute cross-head interactions."""
    he = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    for d in diags:
        for i in range(min(n_layers, len(d.head_energies))):
            for j in range(min(n_heads, len(d.head_energies[i]))):
                he[i][j].append(d.head_energies[i][j])
    return compute_cross_head_cp(he, n_layers, n_heads)


def analyze_model(model_name, label, device, max_tokens, runs, n_bins):
    """Run full CP analysis on a single model across all categories."""
    print(f"\n{'=' * 70}")
    print(f"MODEL: {label} ({model_name})")
    print(f"{'=' * 70}")

    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device).eval()

    probe = FineProbe.from_model(model)
    n_layers = probe.n_layers
    n_heads = probe.n_heads
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {n_layers}L x {n_heads}H = {n_layers * n_heads} heads, {n_params:,} params")

    # Probe each category
    category_diags = {}
    category_reports = {}
    all_diags = []

    for cat_name, prompts in PROMPT_CATEGORIES.items():
        t0 = time.time()
        diags = probe_category(model, tokenizer, probe, prompts, max_tokens, runs)
        elapsed = time.time() - t0
        category_diags[cat_name] = diags
        all_diags.extend(diags)
        print(f"  {cat_name:15s}: {len(diags):5d} tokens  ({elapsed:.1f}s)")

    # Compute CP per category
    for cat_name, diags in category_diags.items():
        category_reports[cat_name] = diags_to_cp(diags, n_layers, n_heads, n_bins)

    # Overall CP
    overall = diags_to_cp(all_diags, n_layers, n_heads, n_bins)

    # Cross-head interactions (overall)
    cross = diags_to_cross(all_diags, n_layers, n_heads)

    # ── Print summary ─────────────────────────────────────────
    print(f"\n  OVERALL:")
    print(f"    Hierarchy:  {overall['hierarchy']}")
    print(f"    Emergence:  {overall['emergence']:.4f}")
    print(f"    S_path:     {overall['S_path_norm']:.4f}")
    print(f"    S_row_bar:  {overall['S_row_bar']:.4f}")
    print(f"    Top head:   L{overall['top_heads'][0]['layer']}H{overall['top_heads'][0]['head']} CP={overall['top_heads'][0]['cp']:.4f}")
    print(f"    Tokens:     {len(all_diags)}")

    print(f"\n  PER-CATEGORY:")
    print(f"  {'Category':15s} {'Hierarchy':15s} {'Emerge':>8s} {'Top Head':>12s} {'TopCP':>8s}")
    print(f"  {'-' * 65}")
    for cat_name in PROMPT_CATEGORIES:
        r = category_reports[cat_name]
        if r.get("status") != "ok":
            print(f"  {cat_name:15s} insufficient data")
            continue
        th = r["top_heads"][0]
        print(f"  {cat_name:15s} {r['hierarchy']:15s} {r['emergence']:8.4f} "
              f"  L{th['layer']}H{th['head']:2d}     {th['cp']:8.4f}")

    # CP shift matrix: for each category pair, which heads change most
    print(f"\n  LAYER CP PROFILE:")
    print(f"  {'Layer':>5s}", end="")
    for cat in PROMPT_CATEGORIES:
        print(f" {cat[:6]:>7s}", end="")
    print()
    for i in range(n_layers):
        print(f"  L{i:3d} ", end="")
        for cat in PROMPT_CATEGORIES:
            r = category_reports[cat]
            if r.get("status") == "ok":
                print(f" {r['layer_cp'][i]:7.4f}", end="")
            else:
                print(f"     N/A", end="")
        print()

    # Biggest CP shifts: halluc vs factual, code vs reasoning, etc.
    print(f"\n  KEY CP SHIFTS:")
    shift_pairs = [
        ("factual", "hallucination", "Halluc vs Factual"),
        ("factual", "code", "Code vs Factual"),
        ("factual", "creative", "Creative vs Factual"),
        ("reasoning", "hallucination", "Halluc vs Reasoning"),
        ("math", "creative", "Creative vs Math"),
        ("instruction", "hallucination", "Halluc vs Instruction"),
    ]

    all_shifts = {}
    for cat_a, cat_b, label_shift in shift_pairs:
        ra = category_reports.get(cat_a, {})
        rb = category_reports.get(cat_b, {})
        if ra.get("status") != "ok" or rb.get("status") != "ok":
            continue

        shifts = []
        for i in range(n_layers):
            for j in range(n_heads):
                delta = rb["head_cp"][i][j] - ra["head_cp"][i][j]
                shifts.append({
                    "layer": i, "head": j, "delta": delta,
                    "cp_a": ra["head_cp"][i][j], "cp_b": rb["head_cp"][i][j],
                })
        shifts.sort(key=lambda x: abs(x["delta"]), reverse=True)
        all_shifts[f"{cat_a}_vs_{cat_b}"] = shifts

        top = shifts[0]
        bot = shifts[-1] if shifts[-1]["delta"] < 0 else shifts[0]
        print(f"    {label_shift:25s}: biggest gain L{top['layer']}H{top['head']} ({top['delta']:+.4f})  "
              f"biggest loss L{bot['layer']}H{bot['head']} ({bot['delta']:+.4f})")

    # Cross-head top interactions
    print(f"\n  TOP CROSS-HEAD LINKS:")
    if cross.get("status") == "ok":
        for ix in cross["top_interactions"][:5]:
            print(f"    L{ix['cause_layer']}H{ix['cause_head']:2d} -> "
                  f"L{ix['effect_layer']}H{ix['effect_head']:2d}  "
                  f"MI={ix['mutual_info']:.4f}")

    # Clean up
    probe.detach()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    gc.collect()

    return {
        "model": model_name,
        "label": label,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_params": n_params,
        "total_tokens": len(all_diags),
        "overall": overall,
        "categories": category_reports,
        "cross_head": {
            "n_interactions": cross.get("n_interactions", 0),
            "top_interactions": cross.get("top_interactions", [])[:30],
        },
        "shifts": {k: v[:20] for k, v in all_shifts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CP Analysis")
    parser.add_argument("--models", nargs="+", default=["gpt2"],
                        choices=["gpt2", "qwen"],
                        help="Model families to test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=2,
                        help="Runs per prompt for averaging")
    parser.add_argument("--bins", type=int, default=16)
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("COMPREHENSIVE CAUSAL PRIMITIVES ANALYSIS")
    print("=" * 70)
    print(f"Models:     {args.models}")
    print(f"Categories: {list(PROMPT_CATEGORIES.keys())}")
    print(f"Device:     {device}")
    print(f"Runs/prompt: {args.runs}")
    print(f"Bins:       {args.bins}")
    print(f"Prompts:    {sum(len(v) for v in PROMPT_CATEGORIES.values())} total")
    print("=" * 70)

    all_results = {}

    for model_family in args.models:
        for model_name, label in MODEL_CONFIGS[model_family]["models"]:
            result = analyze_model(
                model_name, label, device, args.max_tokens, args.runs, args.bins
            )
            all_results[model_name] = result

    # ══════════════════════════════════════════════════════════
    # CROSS-MODEL COMPARISON (if multiple models)
    # ══════════════════════════════════════════════════════════
    model_names = list(all_results.keys())

    if len(model_names) > 1:
        print(f"\n{'=' * 70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'=' * 70}")

        for i, mn_a in enumerate(model_names):
            for mn_b in model_names[i+1:]:
                ra = all_results[mn_a]
                rb = all_results[mn_b]
                print(f"\n  {ra['label']} vs {rb['label']}:")
                print(f"    {'Category':15s} {'Hierarchy A':>15s} {'Hierarchy B':>15s} {'Emerge A':>10s} {'Emerge B':>10s}")

                for cat in PROMPT_CATEGORIES:
                    ca = ra["categories"].get(cat, {})
                    cb = rb["categories"].get(cat, {})
                    if ca.get("status") == "ok" and cb.get("status") == "ok":
                        print(f"    {cat:15s} {ca['hierarchy']:>15s} {cb['hierarchy']:>15s} "
                              f"{ca['emergence']:10.4f} {cb['emergence']:10.4f}")

                # Compare top heads
                print(f"\n    Top 5 causal heads:")
                print(f"    {'Rank':>4s}  {ra['label']:>20s}  {rb['label']:>20s}")
                oa = ra["overall"]
                ob = rb["overall"]
                for rank in range(5):
                    ha = oa["top_heads"][rank] if rank < len(oa["top_heads"]) else None
                    hb = ob["top_heads"][rank] if rank < len(ob["top_heads"]) else None
                    la = f"L{ha['layer']}H{ha['head']} CP={ha['cp']:.4f}" if ha else "N/A"
                    lb = f"L{hb['layer']}H{hb['head']} CP={hb['cp']:.4f}" if hb else "N/A"
                    print(f"    #{rank+1:2d}   {la:>20s}  {lb:>20s}")

    # ── Save results ──────────────────────────────────────────
    data_path = os.path.join(output_dir, "cp_comprehensive_results.json")
    with open(data_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nData saved to: {data_path}")

    # ── Generate plots ────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        for model_name, result in all_results.items():
            n_layers = result["n_layers"]
            n_heads = result["n_heads"]
            cats = list(PROMPT_CATEGORIES.keys())
            n_cats = len(cats)

            fig = plt.figure(figsize=(24, 28))
            safe_label = result["label"].replace("/", "-")
            fig.suptitle(f"Causal Primitives: {result['label']}\n"
                         f"{n_layers}L x {n_heads}H | {result['total_tokens']} tokens",
                         fontsize=14, fontweight="bold")

            gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)

            # Panel 1: Overall CP heatmap
            ax = fig.add_subplot(gs[0, 0])
            cp_map = np.array(result["overall"]["head_cp"])
            im = ax.imshow(cp_map, aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title(f"Overall Head CP — {result['overall']['hierarchy']}")
            plt.colorbar(im, ax=ax, label="CP")

            # Panel 2: Per-category layer CP profiles (stacked)
            ax = fig.add_subplot(gs[0, 1])
            x = np.arange(n_layers)
            w = 0.8 / n_cats
            colors = plt.cm.tab10(np.linspace(0, 1, n_cats))
            for idx, cat in enumerate(cats):
                r = result["categories"].get(cat, {})
                if r.get("status") == "ok":
                    ax.bar(x + idx * w - 0.4, r["layer_cp"], w,
                           label=cat[:8], color=colors[idx], alpha=0.7)
            ax.set_xlabel("Layer"); ax.set_ylabel("CP")
            ax.set_title("Layer CP by Category")
            ax.legend(fontsize=6, ncol=2)

            # Panel 3-4: CP heatmaps for factual vs hallucination
            for panel_idx, cat in enumerate(["factual", "hallucination"]):
                ax = fig.add_subplot(gs[1, panel_idx])
                r = result["categories"].get(cat, {})
                if r.get("status") == "ok":
                    im = ax.imshow(np.array(r["head_cp"]), aspect="auto",
                                   cmap="viridis", interpolation="nearest")
                    plt.colorbar(im, ax=ax, label="CP")
                ax.set_xlabel("Head"); ax.set_ylabel("Layer")
                ax.set_title(f"{cat.title()} Head CP")

            # Panel 5-6: CP heatmaps for code vs creative
            for panel_idx, cat in enumerate(["code", "creative"]):
                ax = fig.add_subplot(gs[2, panel_idx])
                r = result["categories"].get(cat, {})
                if r.get("status") == "ok":
                    im = ax.imshow(np.array(r["head_cp"]), aspect="auto",
                                   cmap="viridis", interpolation="nearest")
                    plt.colorbar(im, ax=ax, label="CP")
                ax.set_xlabel("Head"); ax.set_ylabel("Layer")
                ax.set_title(f"{cat.title()} Head CP")

            # Panel 7: CP shift heatmap (halluc - factual)
            ax = fig.add_subplot(gs[3, 0])
            rf = result["categories"].get("factual", {})
            rh = result["categories"].get("hallucination", {})
            if rf.get("status") == "ok" and rh.get("status") == "ok":
                shift = np.array(rh["head_cp"]) - np.array(rf["head_cp"])
                vmax = max(abs(shift.min()), abs(shift.max())) or 0.01
                im = ax.imshow(shift, aspect="auto", cmap="RdBu_r",
                               interpolation="nearest", vmin=-vmax, vmax=vmax)
                plt.colorbar(im, ax=ax, label="ΔCP")
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title("CP Shift: Hallucination − Factual")

            # Panel 8: CP shift (code - creative)
            ax = fig.add_subplot(gs[3, 1])
            rc = result["categories"].get("code", {})
            rcr = result["categories"].get("creative", {})
            if rc.get("status") == "ok" and rcr.get("status") == "ok":
                shift = np.array(rc["head_cp"]) - np.array(rcr["head_cp"])
                vmax = max(abs(shift.min()), abs(shift.max())) or 0.01
                im = ax.imshow(shift, aspect="auto", cmap="RdBu_r",
                               interpolation="nearest", vmin=-vmax, vmax=vmax)
                plt.colorbar(im, ax=ax, label="ΔCP")
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title("CP Shift: Code − Creative")

            # Panel 9: Emergence comparison across categories
            ax = fig.add_subplot(gs[4, 0])
            cat_names = []
            emergences = []
            for cat in cats:
                r = result["categories"].get(cat, {})
                if r.get("status") == "ok":
                    cat_names.append(cat[:8])
                    emergences.append(r["emergence"])
            ax.bar(range(len(cat_names)), emergences, color="steelblue", alpha=0.8)
            ax.set_xticks(range(len(cat_names)))
            ax.set_xticklabels(cat_names, rotation=45, fontsize=8)
            ax.set_ylabel("Emergent Complexity")
            ax.set_title("Emergence by Category")

            # Panel 10: Summary text
            ax = fig.add_subplot(gs[4, 1])
            ax.axis("off")
            lines = [
                f"Model: {result['label']}",
                f"Params: {result['n_params']:,}",
                f"Tokens: {result['total_tokens']}",
                "",
                f"Overall hierarchy: {result['overall']['hierarchy']}",
                f"Overall emergence: {result['overall']['emergence']:.4f}",
                f"S_path: {result['overall']['S_path_norm']:.4f}",
                f"S_row_bar: {result['overall']['S_row_bar']:.4f}",
                "",
                "Top 5 causal heads:",
            ]
            for h in result["overall"]["top_heads"][:5]:
                lines.append(f"  L{h['layer']}H{h['head']}: CP={h['cp']:.4f} det={h['determinism']:.4f}")
            ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                    fontsize=9, fontfamily="monospace", verticalalignment="top")

            plot_path = os.path.join(output_dir, f"cp_{safe_label.replace(' ', '_').lower()}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {plot_path}")
            plt.close()

        # Cross-model comparison plot (if multiple)
        if len(model_names) > 1:
            fig, axes = plt.subplots(2, len(cats), figsize=(3 * len(cats), 8))
            fig.suptitle("Cross-Model CP Comparison by Category", fontsize=14, fontweight="bold")

            for model_idx, mn in enumerate(model_names[:2]):
                r = all_results[mn]
                for cat_idx, cat in enumerate(cats):
                    ax = axes[model_idx, cat_idx]
                    cr = r["categories"].get(cat, {})
                    if cr.get("status") == "ok":
                        im = ax.imshow(np.array(cr["head_cp"]), aspect="auto",
                                       cmap="viridis", interpolation="nearest")
                    ax.set_title(f"{cat[:6]}" if model_idx == 0 else "", fontsize=8)
                    if cat_idx == 0:
                        ax.set_ylabel(r["label"][:15], fontsize=8)
                    ax.tick_params(labelsize=6)

            plot_path = os.path.join(output_dir, "cp_cross_model_comparison.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Cross-model plot saved: {plot_path}")
            plt.close()

    except ImportError:
        print("matplotlib not available — skipping plots")

    print("\nDone.")


if __name__ == "__main__":
    main()
