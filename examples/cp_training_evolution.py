#!/usr/bin/env python3
"""
Causal Primitives Training Evolution — watch a model learn to think.
====================================================================

Uses EleutherAI's Pythia checkpoints (154 snapshots from random init to
fully trained) to track how causal structure evolves during pretraining.

Each checkpoint is a frozen snapshot of Pythia-1.4B at a specific training
step. We run CP analysis on each, then plot the evolution of:
  - Which heads become causally significant (and when)
  - How the causal hierarchy forms (bottom-heavy → distributed → ?)
  - When cross-head pathways emerge
  - Emergent complexity over training
  - Whether causal structure stabilizes or keeps shifting

This is pretraining archaeology — we're watching the birth of cognition.

38 checkpoints sampled from 154 available:
  - All 11 early checkpoints (step0 through step512, log-spaced)
  - 27 later checkpoints (step1000 through step143000, ~every 5000)

Usage:
    python cp_training_evolution.py                          # Default: Pythia-1.4B, 38 checkpoints
    python cp_training_evolution.py --size 410m              # Smaller model, faster
    python cp_training_evolution.py --size 1.4b --device mps # Apple Silicon
    python cp_training_evolution.py --size 2.8b --device cuda # GPU
    python cp_training_evolution.py --checkpoints 10         # Fewer checkpoints (faster)
"""

import sys, os, json, argparse, time, gc, math

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux import FineProbe
from dflux.causal_primitives import CausalPrimitives, CPConfig, compute_cross_head_cp


# ═══════════════════════════════════════════════════════════
# PYTHIA CHECKPOINT MAP
# ═══════════════════════════════════════════════════════════

# All 154 Pythia checkpoints
EARLY_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
LATE_STEPS = list(range(1000, 144000, 1000))  # 1000 to 143000

ALL_STEPS = EARLY_STEPS + LATE_STEPS  # 154 total

# 38-checkpoint sampling: all early + every ~5000 in late
SAMPLE_38 = (
    EARLY_STEPS +  # 11 early (where structure is born)
    [1000, 3000, 5000, 8000, 12000, 16000, 20000,  # 7 early-mid
     25000, 30000, 35000, 40000, 50000, 60000,       # 6 mid
     70000, 80000, 90000, 100000, 110000,             # 5 late-mid
     120000, 130000, 135000, 140000, 143000]          # 5 late + final
)  # 11 + 27 = 38 checkpoints

# Smaller samples for faster runs
SAMPLE_20 = (
    [0, 1, 4, 16, 64, 256, 512] +
    [1000, 5000, 10000, 20000, 40000, 60000,
     80000, 100000, 120000, 135000, 140000, 142000, 143000]
)

SAMPLE_10 = (
    [0, 4, 64, 512] +
    [5000, 20000, 50000, 100000, 130000, 143000]
)

# Model sizes available
PYTHIA_MODELS = {
    "70m": "EleutherAI/pythia-70m",
    "160m": "EleutherAI/pythia-160m",
    "410m": "EleutherAI/pythia-410m",
    "1b": "EleutherAI/pythia-1b",
    "1.4b": "EleutherAI/pythia-1.4b",
    "2.8b": "EleutherAI/pythia-2.8b",
    "6.9b": "EleutherAI/pythia-6.9b",
    "12b": "EleutherAI/pythia-12b",
}

# Deduped versions (trained on deduplicated data — cleaner experiment)
PYTHIA_DEDUPED = {k: v + "-deduped" for k, v in PYTHIA_MODELS.items()}


# ═══════════════════════════════════════════════════════════
# PROMPT LIBRARY (same as comprehensive test)
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
    "code": [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "# Python function to reverse a linked list\ndef reverse_list(head):\n",
        "import numpy as np\n\n# Compute the eigenvalues of a matrix\ndef eigenvalues(matrix):\n",
        "# Binary search implementation\ndef binary_search(arr, target):\n",
    ],
}


# ═══════════════════════════════════════════════════════════
# PROBING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def probe_checkpoint(model_name, step, device, prompts, max_tokens=48):
    """Load one checkpoint, probe it, return CP results."""
    revision = f"step{step}"

    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=dtype
        )
        model = model.to(device).eval()
    except Exception as e:
        return {"status": "error", "error": str(e), "step": step}

    probe = FineProbe.from_model(model)
    n_layers = probe.n_layers
    n_heads = probe.n_heads

    # Run all prompts
    all_diags = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids)
        probe.reset()
        with torch.no_grad():
            try:
                output = model.generate(
                    input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_tokens, do_sample=True,
                    temperature=0.8, top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            except Exception:
                # Early checkpoints might produce garbage — that's fine
                pass
        all_diags.extend(probe.diagnostics)

    if len(all_diags) < 20:
        probe.detach()
        del model
        gc.collect()
        return {"status": "insufficient_data", "step": step, "n_tokens": len(all_diags)}

    # Compute CP
    cp = CausalPrimitives(n_layers, n_heads, CPConfig(n_bins=16))
    raw_he = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    for d in all_diags:
        he = d.head_energies
        le = [sum(heads) for heads in he] if he else []
        cp.observe_token(he, le, d.J)
        for i in range(min(n_layers, len(he))):
            for j in range(min(n_heads, len(he[i]))):
                raw_he[i][j].append(he[i][j])

    report = cp.compute()
    cross = compute_cross_head_cp(raw_he, n_layers, n_heads)

    # Per-category CP (quick version — factual vs hallucination only)
    cat_reports = {}
    for cat_name, cat_prompts in [("factual", prompts[:4]),
                                   ("hallucination", prompts[4:8]),
                                   ("reasoning", prompts[8:12]),
                                   ("code", prompts[12:16])]:
        cat_diags = []
        for prompt in cat_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids)
            probe.reset()
            with torch.no_grad():
                try:
                    model.generate(
                        input_ids, attention_mask=attention_mask,
                        max_new_tokens=max_tokens, do_sample=True,
                        temperature=0.8, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                except Exception:
                    pass
            cat_diags.extend(probe.diagnostics)

        if len(cat_diags) >= 20:
            cat_cp = CausalPrimitives(n_layers, n_heads, CPConfig(n_bins=16))
            for d in cat_diags:
                he = d.head_energies
                le = [sum(heads) for heads in he] if he else []
                cat_cp.observe_token(he, le, d.J)
            cat_reports[cat_name] = cat_cp.compute()

    # Clean up
    probe.detach()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    gc.collect()

    return {
        "status": "ok",
        "step": step,
        "n_tokens": len(all_diags),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "overall": report,
        "categories": cat_reports,
        "cross_head": {
            "n_interactions": cross.get("n_interactions", 0),
            "top_interactions": cross.get("top_interactions", [])[:15],
        },
    }


# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_evolution(results, model_name, output_dir):
    """Generate training evolution plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available — skipping plots")
        return

    # Filter to successful checkpoints
    ok = [r for r in results if r.get("status") == "ok"]
    if len(ok) < 3:
        print("Not enough successful checkpoints for plotting")
        return

    steps = [r["step"] for r in ok]
    n_layers = ok[0]["n_layers"]
    n_heads = ok[0]["n_heads"]

    # ── Extract time series ──
    emergences = [r["overall"]["emergence"] for r in ok]
    s_paths = [r["overall"].get("S_path_norm", 0) for r in ok]
    s_row_bars = [r["overall"].get("S_row_bar", 0) for r in ok]
    hierarchies = [r["overall"]["hierarchy"] for r in ok]

    # Top head CP over time
    top_cps = [r["overall"]["top_heads"][0]["cp"] if r["overall"].get("top_heads") else 0 for r in ok]

    # Top head identity over time
    top_head_ids = []
    for r in ok:
        th = r["overall"].get("top_heads", [{}])[0]
        top_head_ids.append(f"L{th.get('layer', '?')}H{th.get('head', '?')}")

    # Layer CP profiles over time → heatmap
    layer_cp_over_time = []
    for r in ok:
        lcp = r["overall"].get("layer_cp", [0] * n_layers)
        layer_cp_over_time.append(lcp)
    layer_cp_matrix = np.array(layer_cp_over_time).T  # [n_layers, n_checkpoints]

    # Per-head CP at final checkpoint → compare with step 0
    final_cp = np.array(ok[-1]["overall"]["head_cp"]) if ok[-1]["overall"].get("head_cp") else None
    first_cp = np.array(ok[0]["overall"]["head_cp"]) if ok[0]["overall"].get("head_cp") else None

    # Cross-head top MI over time
    top_mis = []
    for r in ok:
        interactions = r.get("cross_head", {}).get("top_interactions", [])
        top_mis.append(interactions[0]["mutual_info"] if interactions else 0)

    # Category-specific emergence over time
    cat_emergences = {cat: [] for cat in ["factual", "hallucination", "reasoning", "code"]}
    for r in ok:
        for cat in cat_emergences:
            cr = r.get("categories", {}).get(cat, {})
            cat_emergences[cat].append(cr.get("emergence", 0))

    # ── Plot ──
    fig = plt.figure(figsize=(28, 36))
    fig.suptitle(
        f"Causal Structure Evolution During Training\n"
        f"{model_name} | {n_layers}L × {n_heads}H | {len(ok)} checkpoints",
        fontsize=16, fontweight="bold"
    )
    gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3)

    # Use log scale for x-axis since early steps are log-spaced
    def setup_xaxis(ax):
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("Training Step")
        ax.axvline(x=512, color="gray", alpha=0.3, linestyle="--", label="End of log-phase")

    # Panel 1: Emergent complexity over training
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, emergences, "o-", color="steelblue", markersize=4, linewidth=1.5)
    setup_xaxis(ax)
    ax.set_ylabel("Emergent Complexity")
    ax.set_title("Emergence Over Training")
    ax.grid(True, alpha=0.3)

    # Panel 2: S_path and S_row_bar
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, s_paths, "o-", color="coral", markersize=3, linewidth=1.5, label="S_path (spread)")
    ax.plot(steps, s_row_bars, "s-", color="mediumpurple", markersize=3, linewidth=1.5, label="S_row_bar (differentiation)")
    setup_xaxis(ax)
    ax.set_ylabel("Value")
    ax.set_title("Path Entropy & Row Negentropy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Top head CP over training
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, top_cps, "o-", color="forestgreen", markersize=4, linewidth=1.5)
    # Annotate which head is top at key points
    for i in range(0, len(ok), max(1, len(ok) // 8)):
        ax.annotate(top_head_ids[i], (steps[i], top_cps[i]),
                    fontsize=6, rotation=30, ha="left", va="bottom")
    setup_xaxis(ax)
    ax.set_ylabel("Top Head CP")
    ax.set_title("Strongest Causal Head Over Training")
    ax.grid(True, alpha=0.3)

    # Panel 4: Cross-head top MI over training
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, top_mis, "o-", color="darkorange", markersize=4, linewidth=1.5)
    setup_xaxis(ax)
    ax.set_ylabel("Top Cross-Head MI")
    ax.set_title("Strongest Cross-Head Pathway Over Training")
    ax.grid(True, alpha=0.3)

    # Panel 5: Layer CP heatmap over training
    ax = fig.add_subplot(gs[2, :])
    if layer_cp_matrix.shape[1] > 0:
        im = ax.imshow(layer_cp_matrix, aspect="auto", cmap="viridis",
                       interpolation="nearest")
        # X-axis: checkpoint indices with step labels
        n_labels = min(15, len(steps))
        label_indices = np.linspace(0, len(steps) - 1, n_labels, dtype=int)
        ax.set_xticks(label_indices)
        ax.set_xticklabels([f"{steps[i]//1000}k" if steps[i] >= 1000
                            else str(steps[i]) for i in label_indices],
                           fontsize=7, rotation=45)
        ax.set_ylabel("Layer")
        ax.set_xlabel("Training Step")
        ax.set_title("Layer CP Over Training (where does causation live?)")
        plt.colorbar(im, ax=ax, label="Layer CP", shrink=0.6)

    # Panel 6: Hierarchy classification over training
    ax = fig.add_subplot(gs[3, 0])
    hierarchy_map = {"no_signal": 0, "bottom-heavy": 1, "distributed": 2,
                     "mesoscale-peaked": 3, "top-heavy": 4}
    h_values = [hierarchy_map.get(h, -1) for h in hierarchies]
    ax.scatter(steps, h_values, c="steelblue", s=40, zorder=5)
    ax.plot(steps, h_values, "-", color="steelblue", alpha=0.4)
    ax.set_yticks(list(hierarchy_map.values()))
    ax.set_yticklabels(list(hierarchy_map.keys()), fontsize=8)
    setup_xaxis(ax)
    ax.set_title("Causal Hierarchy Classification Over Training")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 7: Per-category emergence over training
    ax = fig.add_subplot(gs[3, 1])
    colors_cat = {"factual": "steelblue", "hallucination": "coral",
                  "reasoning": "forestgreen", "code": "mediumpurple"}
    for cat, vals in cat_emergences.items():
        if any(v > 0 for v in vals):
            ax.plot(steps, vals, "o-", color=colors_cat[cat], markersize=3,
                    linewidth=1.5, label=cat)
    setup_xaxis(ax)
    ax.set_ylabel("Emergence")
    ax.set_title("Category-Specific Emergence Over Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 8-9: Head CP heatmap — first vs last checkpoint
    for panel_idx, (ckpt_idx, title) in enumerate([
        (0, f"Step {steps[0]} (start)"),
        (-1, f"Step {steps[-1]} (final)")
    ]):
        ax = fig.add_subplot(gs[4, panel_idx])
        cp_data = ok[ckpt_idx]["overall"].get("head_cp")
        if cp_data:
            im = ax.imshow(np.array(cp_data), aspect="auto", cmap="viridis",
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, label="CP")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"Head CP — {title}")

    # Panel 10: CP delta (final - first)
    ax = fig.add_subplot(gs[5, 0])
    if final_cp is not None and first_cp is not None:
        delta = final_cp - first_cp
        vmax = max(abs(delta.min()), abs(delta.max())) or 0.01
        im = ax.imshow(delta, aspect="auto", cmap="RdBu_r",
                       interpolation="nearest", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="ΔCP")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title(f"CP Change: Step {steps[-1]} − Step {steps[0]}")

    # Panel 11: Summary text
    ax = fig.add_subplot(gs[5, 1])
    ax.axis("off")

    # Find when emergence first exceeds 50% of final
    final_e = emergences[-1] if emergences else 0
    half_e_step = "N/A"
    for i, e in enumerate(emergences):
        if e >= final_e * 0.5:
            half_e_step = steps[i]
            break

    # Find when top head stabilizes (same head for 3+ consecutive checkpoints)
    stable_step = "never"
    for i in range(2, len(top_head_ids)):
        if top_head_ids[i] == top_head_ids[i-1] == top_head_ids[i-2]:
            stable_step = steps[i-2]
            break

    lines = [
        f"Model: {model_name}",
        f"Architecture: {n_layers}L × {n_heads}H",
        f"Checkpoints analyzed: {len(ok)} / {len(results)}",
        "",
        f"Final hierarchy: {hierarchies[-1]}",
        f"Final emergence: {emergences[-1]:.6f}",
        f"Final top head: {top_head_ids[-1]} (CP={top_cps[-1]:.4f})",
        "",
        f"50% emergence reached at step: {half_e_step}",
        f"Top head stabilized at step: {stable_step}",
        "",
        "Head identity progression:",
    ]
    # Show top head at ~8 milestones
    for i in range(0, len(ok), max(1, len(ok) // 8)):
        lines.append(f"  step {steps[i]:>6d}: {top_head_ids[i]} CP={top_cps[i]:.4f}")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, fontfamily="monospace", verticalalignment="top")

    safe_name = model_name.replace("/", "-")
    plot_path = os.path.join(output_dir, f"cp_evolution_{safe_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def get_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Pythia Training Evolution CP Analysis")
    parser.add_argument("--size", default="1.4b",
                        choices=list(PYTHIA_MODELS.keys()),
                        help="Pythia model size (default: 1.4b)")
    parser.add_argument("--deduped", action="store_true",
                        help="Use deduped variant (cleaner experiment)")
    parser.add_argument("--checkpoints", type=int, default=38,
                        choices=[10, 20, 38],
                        help="Number of checkpoints to sample (10, 20, or 38)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=48,
                        help="Tokens to generate per prompt")
    parser.add_argument("--bins", type=int, default=16)
    args = parser.parse_args()

    if args.deduped:
        model_name = PYTHIA_DEDUPED[args.size]
    else:
        model_name = PYTHIA_MODELS[args.size]

    device = get_device(args.device)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Select checkpoint sample
    if args.checkpoints == 10:
        sample = SAMPLE_10
    elif args.checkpoints == 20:
        sample = SAMPLE_20
    else:
        sample = SAMPLE_38

    # Flatten all prompts
    all_prompts = []
    for cat_prompts in PROMPT_CATEGORIES.values():
        all_prompts.extend(cat_prompts)

    print("=" * 70)
    print("CAUSAL PRIMITIVES TRAINING EVOLUTION")
    print("=" * 70)
    print(f"Model:       {model_name}")
    print(f"Checkpoints: {len(sample)} (of 154 available)")
    print(f"Steps:       {sample[0]} → {sample[-1]}")
    print(f"Device:      {device}")
    print(f"Prompts:     {len(all_prompts)} ({len(PROMPT_CATEGORIES)} categories)")
    print(f"Max tokens:  {args.max_tokens}")
    print("=" * 70)

    results = []

    for idx, step in enumerate(sample):
        t0 = time.time()
        print(f"\n[{idx+1:2d}/{len(sample)}] Step {step:>6d}...", end="", flush=True)

        result = probe_checkpoint(
            model_name, step, device, all_prompts, args.max_tokens
        )
        result["checkpoint_index"] = idx
        results.append(result)

        elapsed = time.time() - t0

        if result["status"] == "ok":
            ov = result["overall"]
            th = ov["top_heads"][0] if ov.get("top_heads") else {}
            print(f"  {result['n_tokens']:4d} tok  {elapsed:5.1f}s  "
                  f"emerge={ov['emergence']:.4f}  "
                  f"hierarchy={ov['hierarchy']}  "
                  f"top=L{th.get('layer', '?')}H{th.get('head', '?')}"
                  f"(CP={th.get('cp', 0):.4f})")
        elif result["status"] == "insufficient_data":
            print(f"  {result.get('n_tokens', 0):4d} tok  {elapsed:5.1f}s  (insufficient data)")
        else:
            print(f"  ERROR: {result.get('error', 'unknown')[:60]}  {elapsed:5.1f}s")

    # ── Summary ──
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{'=' * 70}")
    print(f"EVOLUTION SUMMARY: {model_name}")
    print(f"{'=' * 70}")
    print(f"  Successful checkpoints: {len(ok)} / {len(sample)}")

    if ok:
        print(f"\n  EMERGENCE TRAJECTORY:")
        for r in ok:
            step = r["step"]
            ov = r["overall"]
            th = ov["top_heads"][0] if ov.get("top_heads") else {}
            bar = "█" * int(ov["emergence"] * 5000)  # Visual bar
            print(f"    step {step:>6d}: {ov['emergence']:.6f} {bar:30s} "
                  f"{ov['hierarchy']:15s} L{th.get('layer', '?')}H{th.get('head', '?')}")

        # When does structure emerge?
        emergences = [r["overall"]["emergence"] for r in ok]
        steps_ok = [r["step"] for r in ok]
        final_e = emergences[-1]

        for pct in [0.1, 0.25, 0.5, 0.75, 0.9]:
            threshold = final_e * pct
            for i, e in enumerate(emergences):
                if e >= threshold:
                    print(f"\n    {int(pct*100)}% of final emergence reached at step {steps_ok[i]}")
                    break

    # ── Save ──
    data_path = os.path.join(output_dir, f"cp_evolution_{model_name.replace('/', '-')}.json")
    with open(data_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved: {data_path}")

    # ── Plot ──
    plot_evolution(results, model_name, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
