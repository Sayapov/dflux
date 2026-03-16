#!/usr/bin/env python3
"""
Multi-Scale Telemetry — capture and visualize raw signals at every scale.

Usage:
    python examples/multiscale_telemetry.py --model EleutherAI/pythia-1.4b
    python examples/multiscale_telemetry.py --model gpt2 --prompt "The meaning of life is"
    python examples/multiscale_telemetry.py --model EleutherAI/pythia-1.4b --device mps

Output:
    - telemetry_{model_name}.json   — raw data
    - telemetry_{model_name}.png    — 12-panel visualization
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

# Add parent to path for local dev
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dflux.multiscale_telemetry import MultiScaleTelemetry, TelemetryConfig


def load_model(model_name: str, device: str):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # float32 for accurate telemetry
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded: {n_params:.0f}M params on {device}")
    return model, tokenizer


def run_telemetry(model, tokenizer, prompt: str, max_tokens: int, device: str):
    """Attach telemetry, generate, return data."""
    cfg = TelemetryConfig(
        logit_lens=True,
        logit_lens_top_k=5,
        cross_layer=True,
        mlp_internals=True,
        entropy_cascade=True,
        outlier_detection=True,
    )

    telem = MultiScaleTelemetry.from_model(model, tokenizer, cfg=cfg)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    print(f"\nGenerating {max_tokens} tokens from: '{prompt[:60]}...'")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Output: {generated[:200]}...")

    print(f"\nCaptured {len(telem.snapshots)} token snapshots")
    print(telem.summary())

    return telem


def visualize(telem, model_name: str, output_path: str):
    """16-panel visualization of all telemetry signals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    agg = telem.aggregate()
    if not agg:
        print("No data to visualize.")
        return

    n_layers = agg["n_layers"]
    n_tokens = agg["n_tokens"]
    layers = list(range(n_layers))

    fig, axes = plt.subplots(4, 4, figsize=(32, 20))
    fig.suptitle(
        f"Multi-Scale Telemetry: {model_name}\n{n_tokens} tokens, {n_layers} layers",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Residual Norms ──
    ax = axes[0, 0]
    if "residual_norms_mean" in agg:
        mean = agg["residual_norms_mean"]
        std = agg["residual_norms_std"]
        ax.fill_between(layers, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)], alpha=0.3, color="steelblue")
        ax.plot(layers, mean, "o-", color="steelblue", markersize=3, linewidth=1.5)
    ax.set_title("Residual Stream Norm")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Layer Deltas ──
    ax = axes[0, 1]
    if "residual_deltas_mean" in agg:
        mean = agg["residual_deltas_mean"]
        std = agg["residual_deltas_std"]
        ax.fill_between(layers, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)], alpha=0.3, color="coral")
        ax.plot(layers, mean, "o-", color="coral", markersize=3, linewidth=1.5)
    ax.set_title("Per-Layer Delta (||out - in||)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Delta Norm")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Direction Changes ──
    ax = axes[0, 2]
    if "direction_changes_mean" in agg:
        mean = agg["direction_changes_mean"]
        x = list(range(1, len(mean) + 1))
        ax.bar(x, mean, color="mediumpurple", alpha=0.8)
    ax.set_title("Residual Direction Change (1 - cos_sim)")
    ax.set_xlabel("Layer Transition")
    ax.set_ylabel("Cosine Distance")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Logit Lens Entropy ──
    ax = axes[1, 0]
    if "logit_lens_entropy_mean" in agg:
        mean = agg["logit_lens_entropy_mean"]
        std = agg["logit_lens_entropy_std"]
        ax.fill_between(layers, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)], alpha=0.3, color="forestgreen")
        ax.plot(layers, mean, "o-", color="forestgreen", markersize=3, linewidth=1.5)
    ax.set_title("Logit Lens: Prediction Entropy")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (nats)")
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Top-1 Confidence ──
    ax = axes[1, 1]
    if "logit_lens_top1_prob_mean" in agg:
        mean = agg["logit_lens_top1_prob_mean"]
        ax.plot(layers, mean, "o-", color="darkgreen", markersize=3, linewidth=1.5)
        ax.set_yscale("log")
    ax.set_title("Logit Lens: Top-1 Token Probability")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability (log)")
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Entropy Reduction Rate ──
    ax = axes[1, 2]
    if "entropy_reduction_mean" in agg:
        mean = agg["entropy_reduction_mean"]
        x = list(range(1, len(mean) + 1))
        colors = ["green" if v > 0 else "red" for v in mean]
        ax.bar(x, mean, color=colors, alpha=0.8)
    ax.set_title("Entropy Reduction Per Layer")
    ax.set_xlabel("Layer Transition")
    ax.set_ylabel("ΔEntropy (positive = more certain)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 7: MLP vs Attention Energy ──
    ax = axes[2, 0]
    if "mlp_norms_mean" in agg and "attn_norms_mean" in agg:
        mlp = agg["mlp_norms_mean"]
        attn = agg["attn_norms_mean"]
        width = 0.35
        ax.bar([l - width / 2 for l in layers], attn, width, label="Attention", color="royalblue", alpha=0.8)
        ax.bar([l + width / 2 for l in layers], mlp, width, label="MLP", color="orangered", alpha=0.8)
        ax.legend()
    ax.set_title("Attention vs MLP Output Norms")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 8: MLP / Total Ratio ──
    ax = axes[2, 1]
    if "mlp_attn_ratio_mean" in agg:
        mean = agg["mlp_attn_ratio_mean"]
        ax.bar(layers, mean, color="darkorange", alpha=0.8)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="50/50")
        ax.legend()
    ax.set_title("MLP Energy Fraction (MLP / Total)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 9: Cross-Layer Similarity Matrix ──
    ax = axes[2, 2]
    if "cross_layer_sim_mean" in agg:
        sim = np.array(agg["cross_layer_sim_mean"])
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Cross-Layer Delta Similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")

    # ── Panel 10: Activation Gini Coefficient ──
    ax = axes[3, 0]
    if "outlier_gini_mean" in agg:
        mean = agg["outlier_gini_mean"]
        ax.plot(layers, mean, "o-", color="darkred", markersize=3, linewidth=1.5)
    ax.set_title("Activation Gini (inequality of dim magnitudes)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gini (0=equal, 1=single dim)")
    ax.grid(True, alpha=0.3)

    # ── Panel 11: Outlier Max Magnitude ──
    ax = axes[3, 1]
    if "outlier_max_magnitude_mean" in agg:
        mean = agg["outlier_max_magnitude_mean"]
        std = agg["outlier_max_magnitude_std"]
        ax.fill_between(layers, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)], alpha=0.3, color="crimson")
        ax.plot(layers, mean, "o-", color="crimson", markersize=3, linewidth=1.5)
    ax.set_title("Max Activation Magnitude (outlier dims)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max |activation|")
    ax.grid(True, alpha=0.3)

    # ── Panel 12: MLP Dead Neuron Fraction ──
    ax = axes[3, 2]
    if "mlp_dead_frac_mean" in agg:
        mean = agg["mlp_dead_frac_mean"]
        ax.bar(layers, mean, color="gray", alpha=0.8)
    ax.set_title("MLP Dead Neuron Fraction (|act| < 1e-6)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction Dead")
    ax.grid(True, alpha=0.3, axis="y")

    # ══════════════════════════════════════════════════════════
    # DILUTION ANALYSIS (Column 4) — inspired by Moonshot AttnRes
    # ══════════════════════════════════════════════════════════

    # ── Panel 13: Dilution Survival ──
    ax = axes[0, 3]
    if "dilution_survival_mean" in agg:
        mean = agg["dilution_survival_mean"]
        std = agg["dilution_survival_std"]
        colors = ["green" if v > 0 else "red" for v in mean]
        ax.bar(layers, mean, color=colors, alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Dilution: Layer Survival (cos_sim with output)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 14: Dilution Energy Fraction ──
    ax = axes[1, 3]
    if "dilution_energy_frac_mean" in agg:
        mean = agg["dilution_energy_frac_mean"]
        ax.bar(layers, mean, color="teal", alpha=0.8)
    ax.set_title("Dilution: Energy Fraction Reaching Output")
    ax.set_xlabel("Layer")
    ax.set_ylabel("||proj(δ→h_final)|| / ||h_final||")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 15: Wasted Work ──
    ax = axes[2, 3]
    if "dilution_wasted_work_mean" in agg:
        mean = agg["dilution_wasted_work_mean"]
        ax.bar(layers, mean, color="indianred", alpha=0.8)
    ax.set_title("Dilution: Wasted Work (energy × (1 - |survival|))")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Wasted Energy")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 16: Cumulative Drift ──
    ax = axes[3, 3]
    if "dilution_cumulative_drift_mean" in agg:
        mean = agg["dilution_cumulative_drift_mean"]
        std = agg["dilution_cumulative_drift_std"]
        ax.fill_between(layers, [m - s for m, s in zip(mean, std)],
                        [m + s for m, s in zip(mean, std)], alpha=0.3, color="darkorchid")
        ax.plot(layers, mean, "o-", color="darkorchid", markersize=3, linewidth=1.5)
    ax.set_title("Cumulative Drift from L0 (1 - cos_sim)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Distance from Initial")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved → {output_path}")


def visualize_token_evolution(telem, model_name: str, output_path: str):
    """Heatmap view: signal evolution across tokens × layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(telem.snapshots) < 2:
        print("Need at least 2 tokens for evolution view.")
        return

    n_tokens = len(telem.snapshots)
    n_layers = telem.n_layers

    # Build matrices: [n_tokens, n_layers]
    signals = {}

    # Residual norms
    norms = np.array([s.residual_norms for s in telem.snapshots])
    signals["Residual Norm"] = norms

    # Residual deltas
    deltas = np.array([s.residual_deltas for s in telem.snapshots])
    signals["Layer Delta"] = deltas

    # Logit lens entropy
    if telem.snapshots[0].logit_lens_entropy is not None:
        ent = np.array([s.logit_lens_entropy for s in telem.snapshots])
        signals["Prediction Entropy"] = ent

    # Logit lens top-1 prob
    if telem.snapshots[0].logit_lens_top1_prob is not None:
        p = np.array([s.logit_lens_top1_prob for s in telem.snapshots])
        signals["Top-1 Probability"] = np.log10(p + 1e-10)

    # MLP ratio
    if telem.snapshots[0].mlp_attn_ratio is not None:
        r = np.array([s.mlp_attn_ratio for s in telem.snapshots])
        signals["MLP Energy Fraction"] = r

    # Outlier Gini
    if telem.snapshots[0].outlier_gini is not None:
        g = np.array([s.outlier_gini for s in telem.snapshots])
        signals["Activation Gini"] = g

    # Dilution survival
    if telem.snapshots[0].dilution_survival is not None:
        surv = np.array([s.dilution_survival for s in telem.snapshots])
        signals["Dilution Survival"] = surv

    # Dilution wasted work
    if telem.snapshots[0].dilution_wasted_work is not None:
        wasted = np.array([s.dilution_wasted_work for s in telem.snapshots])
        signals["Wasted Work"] = wasted

    n_panels = len(signals)
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"Token × Layer Evolution: {model_name}\n{n_tokens} tokens, {n_layers} layers",
        fontsize=14, fontweight="bold", y=1.0,
    )

    for ax, (name, matrix) in zip(axes, signals.items()):
        im = ax.imshow(matrix.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("Token Index")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Token evolution saved → {output_path}")


def visualize_dimension_channels(telem, model_name: str, output_path: str):
    """Visualize persistent dimension channels across layers.

    This is the multi-scale view: which hidden dimensions carry energy
    consistently across the entire network, and where do they appear?
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = telem.aggregate()
    if "stable_dimension_channels" not in agg or not agg["stable_dimension_channels"]:
        print("No stable dimension channels found.")
        return

    channels = agg["stable_dimension_channels"]
    heatmap = agg.get("channel_layer_heatmap", {})
    n_layers = agg["n_layers"]

    # Limit to top 30 channels for readability
    channels = channels[:30]
    dim_ids = [ch["dim"] for ch in channels]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        f"Dimension Channels: {model_name}\n"
        f"{len(channels)} stable channels (persistent across >50% of tokens & layers)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Channel energy bar chart ──
    ax = axes[0, 0]
    energies = [ch["mean_energy"] for ch in channels]
    errors = [ch["std_energy"] for ch in channels]
    x = range(len(channels))
    ax.bar(x, energies, yerr=errors, color="steelblue", alpha=0.8, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dim_ids], rotation=45, ha="right", fontsize=7)
    ax.set_title("Stable Channel Energy (mean ± std)")
    ax.set_xlabel("Hidden Dimension Index")
    ax.set_ylabel("Mean Activation Magnitude")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 2: Channel frequency ──
    ax = axes[0, 1]
    freqs = [ch["token_frequency"] for ch in channels]
    ax.barh(range(len(channels)), freqs, color="coral", alpha=0.8)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([f"dim {d}" for d in dim_ids], fontsize=7)
    ax.set_title("Token Frequency (fraction of tokens where dim is in top-20)")
    ax.set_xlabel("Frequency")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # ── Panel 3: Layer presence heatmap ──
    ax = axes[1, 0]
    if heatmap:
        # Build matrix: [n_channels, n_layers]
        matrix = np.zeros((len(dim_ids), n_layers))
        for i, dim_id in enumerate(dim_ids):
            layer_data = heatmap.get(str(dim_id), {})
            for layer_str, freq in layer_data.items():
                matrix[i, int(layer_str)] = freq

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                        interpolation="nearest")
        ax.set_yticks(range(len(dim_ids)))
        ax.set_yticklabels([f"dim {d}" for d in dim_ids], fontsize=7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Hidden Dimension")
        ax.set_title("Channel × Layer Presence (fraction of tokens)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No heatmap data", ha="center", va="center")

    # ── Panel 4: Layer-level channel density ──
    ax = axes[1, 1]
    if heatmap:
        # How many channels are active at each layer?
        layer_density = np.zeros(n_layers)
        for dim_id in dim_ids:
            layer_data = heatmap.get(str(dim_id), {})
            for layer_str, freq in layer_data.items():
                if freq > 0.5:  # Active in >50% of tokens
                    layer_density[int(layer_str)] += 1

        ax.bar(range(n_layers), layer_density, color="mediumpurple", alpha=0.8)
        ax.set_title("Active Stable Channels Per Layer")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Number of stable channels active")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dimension channels saved → {output_path}")


def compare_models(json_paths: list, output_path: str):
    """Side-by-side dilution comparison across multiple models.

    Loads telemetry JSONs and produces a comparison visualization showing
    how dilution scales with model depth and architecture.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = []
    for p in json_paths:
        with open(p) as f:
            data = json.load(f)
        models.append(data)

    n_models = len(models)
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_models, 3)))

    fig, axes = plt.subplots(3, 3, figsize=(27, 18))
    fig.suptitle(
        f"Cross-Model Dilution Comparison — {n_models} Models",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Survival by normalized layer position ──
    ax = axes[0, 0]
    for i, data in enumerate(models):
        agg = data["aggregate"]
        if "dilution_survival_mean" not in agg:
            continue
        surv = agg["dilution_survival_mean"]
        n_layers = agg["n_layers"]
        x = np.linspace(0, 1, n_layers)  # normalize layer position
        label = f'{data["model"]} ({n_layers}L)'
        ax.plot(x, surv, "o-", color=colors[i], markersize=3, linewidth=1.5, label=label)
    ax.set_title("Layer Survival vs Normalized Depth")
    ax.set_xlabel("Relative Layer Position (0=first, 1=last)")
    ax.set_ylabel("Cosine Similarity with Output")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # ── Panel 2: Wasted work by normalized position ──
    ax = axes[0, 1]
    for i, data in enumerate(models):
        agg = data["aggregate"]
        if "dilution_wasted_work_mean" not in agg:
            continue
        wasted = agg["dilution_wasted_work_mean"]
        n_layers = agg["n_layers"]
        # Normalize wasted work by total delta to compare across model sizes
        deltas = agg["residual_deltas_mean"]
        total_delta = sum(deltas)
        norm_wasted = [w / total_delta for w in wasted]
        x = np.linspace(0, 1, n_layers)
        label = f'{data["model"]} ({n_layers}L)'
        ax.plot(x, norm_wasted, "o-", color=colors[i], markersize=3, linewidth=1.5, label=label)
    ax.set_title("Normalized Wasted Work vs Depth")
    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("Wasted / Total Delta Energy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Cumulative drift comparison ──
    ax = axes[0, 2]
    for i, data in enumerate(models):
        agg = data["aggregate"]
        if "dilution_cumulative_drift_mean" not in agg:
            continue
        drift = agg["dilution_cumulative_drift_mean"]
        n_layers = agg["n_layers"]
        x = np.linspace(0, 1, n_layers)
        label = f'{data["model"]} ({n_layers}L)'
        ax.plot(x, drift, "o-", color=colors[i], markersize=3, linewidth=1.5, label=label)
    ax.set_title("Cumulative Drift from L0")
    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("Cosine Distance from Initial")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Energy fraction comparison ──
    ax = axes[1, 0]
    for i, data in enumerate(models):
        agg = data["aggregate"]
        if "dilution_energy_frac_mean" not in agg:
            continue
        efrac = agg["dilution_energy_frac_mean"]
        n_layers = agg["n_layers"]
        x = np.linspace(0, 1, n_layers)
        label = f'{data["model"]} ({n_layers}L)'
        ax.plot(x, efrac, "o-", color=colors[i], markersize=3, linewidth=1.5, label=label)
    ax.set_title("Energy Fraction Reaching Output")
    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("Fraction of Final Output")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Entropy cascade comparison ──
    ax = axes[1, 1]
    for i, data in enumerate(models):
        agg = data["aggregate"]
        if "logit_lens_entropy_mean" not in agg:
            continue
        ent = agg["logit_lens_entropy_mean"]
        n_layers = agg["n_layers"]
        # Normalize entropy to [0, 1] range (fraction of initial)
        if ent[0] > 0:
            norm_ent = [e / ent[0] for e in ent]
        else:
            norm_ent = ent
        x = np.linspace(0, 1, n_layers)
        label = f'{data["model"]} ({n_layers}L)'
        ax.plot(x, norm_ent, "o-", color=colors[i], markersize=3, linewidth=1.5, label=label)
    ax.set_title("Normalized Entropy Cascade")
    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("Entropy / Initial Entropy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Residual norm growth comparison ──
    ax = axes[1, 2]
    for i, data in enumerate(models):
        agg = data["aggregate"]
        if "residual_norms_mean" not in agg:
            continue
        norms = agg["residual_norms_mean"]
        n_layers = agg["n_layers"]
        # Normalize to growth factor from L0
        norm_growth = [n / norms[0] for n in norms]
        x = np.linspace(0, 1, n_layers)
        label = f'{data["model"]} ({n_layers}L)'
        ax.plot(x, norm_growth, "o-", color=colors[i], markersize=3, linewidth=1.5, label=label)
    ax.set_title("Residual Norm Growth (relative to L0)")
    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("Norm / L0 Norm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 7: MLP dominance comparison ──
    ax = axes[2, 0]
    for i, data in enumerate(models):
        agg = data["aggregate"]
        if "mlp_attn_ratio_mean" not in agg:
            continue
        ratio = agg["mlp_attn_ratio_mean"]
        n_layers = agg["n_layers"]
        x = np.linspace(0, 1, n_layers)
        label = f'{data["model"]} ({n_layers}L)'
        ax.plot(x, ratio, "o-", color=colors[i], markersize=3, linewidth=1.5, label=label)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title("MLP Energy Fraction")
    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("MLP / (MLP + Attn)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 8: Summary bar chart ──
    ax = axes[2, 1]
    model_names = []
    waste_ratios = []
    for data in models:
        agg = data["aggregate"]
        if "dilution_wasted_work_mean" not in agg:
            continue
        wasted = sum(agg["dilution_wasted_work_mean"])
        total = sum(agg["residual_deltas_mean"])
        model_names.append(f'{data["model"]}\n({agg["n_layers"]}L)')
        waste_ratios.append(wasted / total if total > 0 else 0)

    if model_names:
        bars = ax.bar(range(len(model_names)), [w * 100 for w in waste_ratios],
                      color=[colors[i] for i in range(len(model_names))], alpha=0.8)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=8)
        ax.set_title("Total Waste Ratio")
        ax.set_ylabel("% of Delta Energy Wasted")
        ax.grid(True, alpha=0.3, axis="y")
        # Add value labels on bars
        for bar, val in zip(bars, waste_ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Panel 9: Stable channels count + norm growth comparison ──
    ax = axes[2, 2]
    model_labels = []
    n_channels_list = []
    norm_growth_list = []
    for data in models:
        agg = data["aggregate"]
        n_ch = agg.get("n_stable_channels", 0)
        norms = agg.get("residual_norms_mean", [1, 1])
        growth = norms[-1] / norms[0] if norms[0] > 0 else 1
        model_labels.append(f'{data["model"]}\n({agg["n_layers"]}L)')
        n_channels_list.append(n_ch)
        norm_growth_list.append(growth)

    if model_labels:
        x_pos = np.arange(len(model_labels))
        width = 0.35
        ax2 = ax.twinx()
        bars1 = ax.bar(x_pos - width / 2, n_channels_list, width,
                        color="steelblue", alpha=0.8, label="Stable Channels")
        bars2 = ax2.bar(x_pos + width / 2, norm_growth_list, width,
                         color="coral", alpha=0.8, label="Norm Growth")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_labels, fontsize=8)
        ax.set_ylabel("# Stable Dimension Channels", color="steelblue")
        ax2.set_ylabel("Norm Growth Factor (L_last / L0)", color="coral")
        ax.set_title("Structural Complexity")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Scale Telemetry")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name (or comma-separated for multi-model)")
    parser.add_argument("--prompt", type=str,
                        default="The theory of everything begins with the observation that",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=48,
                        help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--output-dir", type=str, default="data/telemetry",
                        help="Where to save outputs")
    parser.add_argument("--compare", type=str, default=None,
                        help="Comma-separated list of telemetry JSON paths to compare")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compare mode: load existing JSONs and produce comparison
    if args.compare:
        json_paths = [p.strip() for p in args.compare.split(",")]
        compare_path = out_dir / "telemetry_comparison.png"
        compare_models(json_paths, str(compare_path))
        return

    # Multi-model mode: comma-separated model names
    model_names = [m.strip() for m in args.model.split(",")]
    json_paths = []

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model, tokenizer = load_model(model_name, args.device)
        telem = run_telemetry(model, tokenizer, args.prompt, args.max_tokens, args.device)

        safe_name = model_name.replace("/", "-")

        json_path = out_dir / f"telemetry_{safe_name}.json"
        telem.save(str(json_path))
        json_paths.append(str(json_path))

        png_path = out_dir / f"telemetry_{safe_name}.png"
        visualize(telem, model_name, str(png_path))

        evo_path = out_dir / f"telemetry_{safe_name}_evolution.png"
        visualize_token_evolution(telem, model_name, str(evo_path))

        chan_path = out_dir / f"telemetry_{safe_name}_channels.png"
        visualize_dimension_channels(telem, model_name, str(chan_path))

        telem.detach()

        # Free memory before loading next model
        del model, tokenizer, telem
        import gc
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()

        print(f"\nDone: {model_name}")
        print(f"  Data:       {json_path}")
        print(f"  Aggregate:  {png_path}")
        print(f"  Evolution:  {evo_path}")
        print(f"  Channels:   {chan_path}")

    # Auto-compare if multiple models
    if len(json_paths) > 1:
        compare_path = out_dir / "telemetry_comparison.png"
        compare_models(json_paths, str(compare_path))
        print(f"\n  Comparison: {compare_path}")


if __name__ == "__main__":
    main()
