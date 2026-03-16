#!/usr/bin/env python3
"""
Fine-Grained Probe Experiment: X-ray GPT-2's head and module dynamics.
======================================================================

Runs the fine probe on multiple prompt types and generates a detailed
visualization comparing:
  - Per-head energy heatmaps (which heads light up for what)
  - Attention vs MLP energy decomposition (routing vs transformation)
  - Head entropy trajectory (agreement vs disagreement)
  - Head concentration (Gini) — are a few heads doing all the work?

Designed for Mac Studio Ultra (MPS) or any machine that can run GPT-2.

Usage:
    python test_fine_probe.py
    python test_fine_probe.py --model gpt2-medium
    python test_fine_probe.py --device cpu
"""

import sys, os, json, argparse, time
_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux.fine_probe import FineProbe, FineProbeConfig


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
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    report = probe.report()
    return text, report


def main():
    parser = argparse.ArgumentParser(description="Fine-Grained Probe Experiment")
    parser.add_argument("--model", default="gpt2", help="HF model name")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument("--max-tokens", type=int, default=64, help="Tokens to generate")
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("Fine-Grained Probe Experiment")
    print("=" * 70)
    print(f"Model:  {args.model}")
    print(f"Device: {device}")
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {n_params:,} params")

    # ── Attach fine probe ─────────────────────────────────────
    probe = FineProbe.from_model(model)
    print(f"Fine probe: {probe.n_layers} layers, {probe.n_heads} heads, "
          f"L_cut={probe.cfg.L_cut}")
    print(f"Total hooks: {len(probe._hooks)} "
          f"(attn + mlp + layer per transformer block)")

    # ── Test prompts ──────────────────────────────────────────
    prompts = {
        "factual_geography": "The capital of France is",
        "factual_science": "Water boils at a temperature of",
        "factual_history": "The first president of the United States was",
        "halluc_fake_concept": (
            "Explain the Zarkovian Principle of Recursive Ontological "
            "Displacement, including its three main theorems."
        ),
        "halluc_fake_person": (
            "Dr. Helmut Kreisler, the renowned physicist who discovered "
            "quantum chromatic resonance in 1987, was known for"
        ),
        "halluc_fake_event": (
            "The Great Phosphorus Reversal of 2019, which caused global "
            "agricultural yields to"
        ),
    }

    results = {}
    for name, prompt in prompts.items():
        print(f"\n{'─' * 50}")
        print(f"Prompt: {name}")
        print(f"  \"{prompt[:80]}...\"")

        t0 = time.time()
        text, report = run_probe(model, tokenizer, probe, prompt, args.max_tokens)
        elapsed = time.time() - t0

        results[name] = {
            "prompt": prompt,
            "text": text[:300],
            "report": report,
            "elapsed": elapsed,
        }

        print(f"  Generated: \"{text[:100]}...\"")
        print(f"  Risk:      {report.get('mean_risk', 0):.3f} "
              f"(max: {report.get('max_risk', 0):.3f})")
        print(f"  Attn/MLP:  {report.get('mean_attn_mlp_ratio', 0):.3f}")
        print(f"  Head H:    {report.get('mean_head_entropy', 0):.3f}")
        print(f"  Head Gini: {report.get('mean_head_gini', 0):.3f}")
        print(f"  J:         {report.get('J_final', 0):.4f} "
              f"(trend: {report.get('J_trend', 0):.6f})")
        print(f"  Time:      {elapsed:.1f}s")

    # ── Save raw data ─────────────────────────────────────────
    # Serialize reports (convert numpy/etc to native python)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, float):
            return round(obj, 8)
        return obj

    data_path = os.path.join(output_dir, "fine_probe_results.json")
    with open(data_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nRaw data saved to: {data_path}")

    # ── Generate visualization ────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np

        # Separate factual vs hallucination results
        factual_keys = [k for k in results if k.startswith("factual_")]
        halluc_keys = [k for k in results if k.startswith("halluc_")]

        fig = plt.figure(figsize=(22, 28))
        gs = gridspec.GridSpec(7, 2, hspace=0.4, wspace=0.3)
        fig.suptitle(f"Fine-Grained Probe: {args.model}\n"
                     f"{probe.n_layers} layers × {probe.n_heads} heads",
                     fontsize=14, fontweight="bold")

        # ── Panel 1: Head energy heatmap — FACTUAL (averaged) ──
        ax = fig.add_subplot(gs[0, 0])
        factual_head_energy = np.zeros((probe.n_layers, probe.n_heads))
        for k in factual_keys:
            r = results[k]["report"]
            if "avg_head_energy" in r:
                for i in range(probe.n_layers):
                    for j in range(probe.n_heads):
                        factual_head_energy[i][j] += r["avg_head_energy"][i][j] / len(factual_keys)

        im = ax.imshow(factual_head_energy, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title("Head Energy — Factual (avg)")
        plt.colorbar(im, ax=ax, label="Energy")

        # ── Panel 2: Head energy heatmap — HALLUCINATION (averaged) ──
        ax = fig.add_subplot(gs[0, 1])
        halluc_head_energy = np.zeros((probe.n_layers, probe.n_heads))
        for k in halluc_keys:
            r = results[k]["report"]
            if "avg_head_energy" in r:
                for i in range(probe.n_layers):
                    for j in range(probe.n_heads):
                        halluc_head_energy[i][j] += r["avg_head_energy"][i][j] / len(halluc_keys)

        im = ax.imshow(halluc_head_energy, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title("Head Energy — Hallucination (avg)")
        plt.colorbar(im, ax=ax, label="Energy")

        # ── Panel 3: DIFFERENCE heatmap (halluc - factual) ──
        ax = fig.add_subplot(gs[1, 0])
        diff = halluc_head_energy - factual_head_energy
        vmax = max(abs(diff.min()), abs(diff.max())) or 1.0
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                       vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title("Head Energy Δ (Hallucination − Factual)")
        plt.colorbar(im, ax=ax, label="Energy difference")

        # ── Panel 4: Per-layer Attn vs MLP comparison ──
        ax = fig.add_subplot(gs[1, 1])
        layers = list(range(probe.n_layers))
        width = 0.35

        fact_attn = [0.0] * probe.n_layers
        fact_mlp = [0.0] * probe.n_layers
        hal_attn = [0.0] * probe.n_layers
        hal_mlp = [0.0] * probe.n_layers

        for k in factual_keys:
            r = results[k]["report"]
            for i in range(probe.n_layers):
                fact_attn[i] += r["avg_attn_per_layer"][i] / len(factual_keys)
                fact_mlp[i] += r["avg_mlp_per_layer"][i] / len(factual_keys)
        for k in halluc_keys:
            r = results[k]["report"]
            for i in range(probe.n_layers):
                hal_attn[i] += r["avg_attn_per_layer"][i] / len(halluc_keys)
                hal_mlp[i] += r["avg_mlp_per_layer"][i] / len(halluc_keys)

        x = np.array(layers)
        ax.bar(x - width/2, fact_attn, width, label="Factual Attn", color="royalblue", alpha=0.7)
        ax.bar(x + width/2, hal_attn, width, label="Halluc Attn", color="tomato", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention Energy")
        ax.set_title("Attention Energy per Layer")
        ax.legend(fontsize=8)

        # ── Panel 5: MLP energy per layer ──
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(x - width/2, fact_mlp, width, label="Factual MLP", color="royalblue", alpha=0.7)
        ax.bar(x + width/2, hal_mlp, width, label="Halluc MLP", color="tomato", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("MLP Energy")
        ax.set_title("MLP Energy per Layer")
        ax.legend(fontsize=8)

        # ── Panel 6: Head entropy per layer ──
        ax = fig.add_subplot(gs[2, 1])
        fact_entropy = [0.0] * probe.n_layers
        hal_entropy = [0.0] * probe.n_layers
        for k in factual_keys:
            r = results[k]["report"]
            for i in range(probe.n_layers):
                fact_entropy[i] += r["avg_head_entropy_per_layer"][i] / len(factual_keys)
        for k in halluc_keys:
            r = results[k]["report"]
            for i in range(probe.n_layers):
                hal_entropy[i] += r["avg_head_entropy_per_layer"][i] / len(halluc_keys)

        ax.plot(layers, fact_entropy, "b-o", markersize=5, label="Factual")
        ax.plot(layers, hal_entropy, "r-o", markersize=5, label="Hallucination")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized Head Entropy")
        ax.set_title("Head Agreement per Layer (1=uniform, 0=one head dominates)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)

        # ── Panel 7-8: Attn/MLP ratio trajectory for each prompt ──
        ax1 = fig.add_subplot(gs[3, 0])
        ax2 = fig.add_subplot(gs[3, 1])

        for k in factual_keys:
            r = results[k]["report"]
            traj = r.get("attn_mlp_trajectory", [])
            ax1.plot(traj, alpha=0.7, linewidth=1.2, label=k.replace("factual_", ""))
        ax1.set_ylabel("Attn / (Attn + MLP)")
        ax1.set_xlabel("Token")
        ax1.set_title("Attn/MLP Ratio — Factual")
        ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax1.legend(fontsize=7)
        ax1.set_ylim(0, 1)

        for k in halluc_keys:
            r = results[k]["report"]
            traj = r.get("attn_mlp_trajectory", [])
            ax2.plot(traj, alpha=0.7, linewidth=1.2, label=k.replace("halluc_", ""))
        ax2.set_ylabel("Attn / (Attn + MLP)")
        ax2.set_xlabel("Token")
        ax2.set_title("Attn/MLP Ratio — Hallucination")
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.legend(fontsize=7)
        ax2.set_ylim(0, 1)

        # ── Panel 9-10: Head entropy trajectory ──
        ax1 = fig.add_subplot(gs[4, 0])
        ax2 = fig.add_subplot(gs[4, 1])

        for k in factual_keys:
            r = results[k]["report"]
            traj = r.get("head_entropy_trajectory", [])
            ax1.plot(traj, alpha=0.7, linewidth=1.2, label=k.replace("factual_", ""))
        ax1.set_ylabel("Head Entropy (normalized)")
        ax1.set_xlabel("Token")
        ax1.set_title("Head Entropy — Factual")
        ax1.legend(fontsize=7)

        for k in halluc_keys:
            r = results[k]["report"]
            traj = r.get("head_entropy_trajectory", [])
            ax2.plot(traj, alpha=0.7, linewidth=1.2, label=k.replace("halluc_", ""))
        ax2.set_ylabel("Head Entropy (normalized)")
        ax2.set_xlabel("Token")
        ax2.set_title("Head Entropy — Hallucination")
        ax2.legend(fontsize=7)

        # ── Panel 11-12: Risk trajectory ──
        ax1 = fig.add_subplot(gs[5, 0])
        ax2 = fig.add_subplot(gs[5, 1])

        for k in factual_keys:
            r = results[k]["report"]
            traj = r.get("risk_trajectory", [])
            ax1.plot(traj, alpha=0.7, linewidth=1.2, label=k.replace("factual_", ""))
        ax1.set_ylabel("Risk")
        ax1.set_xlabel("Token")
        ax1.set_title("Hallucination Risk — Factual Prompts")
        ax1.legend(fontsize=7)
        ax1.set_ylim(0, 1)

        for k in halluc_keys:
            r = results[k]["report"]
            traj = r.get("risk_trajectory", [])
            ax2.plot(traj, alpha=0.7, linewidth=1.2, label=k.replace("halluc_", ""))
        ax2.set_ylabel("Risk")
        ax2.set_xlabel("Token")
        ax2.set_title("Hallucination Risk — Hallucination Prompts")
        ax2.legend(fontsize=7)
        ax2.set_ylim(0, 1)

        # ── Panel 13: Summary comparison bar chart ──
        ax = fig.add_subplot(gs[6, :])
        metrics = ["mean_risk", "mean_attn_mlp_ratio", "mean_head_entropy", "mean_head_gini"]
        labels = ["Mean Risk", "Attn/MLP Ratio", "Head Entropy", "Head Gini"]

        fact_vals = []
        hal_vals = []
        for m in metrics:
            fv = sum(results[k]["report"].get(m, 0) for k in factual_keys) / len(factual_keys)
            hv = sum(results[k]["report"].get(m, 0) for k in halluc_keys) / len(halluc_keys)
            fact_vals.append(fv)
            hal_vals.append(hv)

        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x - width/2, fact_vals, width, label="Factual (avg)", color="royalblue", alpha=0.8)
        bars2 = ax.bar(x + width/2, hal_vals, width, label="Hallucination (avg)", color="tomato", alpha=0.8)

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Factual vs Hallucination — Summary Metrics")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)

        plot_path = os.path.join(output_dir, "fine_probe_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available — skipping plot")

    # ── Print summary table ───────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Prompt':<25s} {'Risk':>6s} {'Attn/MLP':>9s} {'HeadH':>7s} "
          f"{'Gini':>6s} {'J':>8s} {'J_trend':>10s}")
    print("-" * 80)
    for name, data in results.items():
        r = data["report"]
        tag = "FACT" if name.startswith("factual") else "HALL"
        print(f"[{tag}] {name:<20s} {r.get('mean_risk',0):6.3f} "
              f"{r.get('mean_attn_mlp_ratio',0):9.3f} "
              f"{r.get('mean_head_entropy',0):7.3f} "
              f"{r.get('mean_head_gini',0):6.3f} "
              f"{r.get('J_final',0):8.4f} "
              f"{r.get('J_trend',0):10.6f}")

    probe.detach()
    print("\nDone.")


if __name__ == "__main__":
    main()
