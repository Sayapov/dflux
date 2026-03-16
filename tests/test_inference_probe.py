#!/usr/bin/env python3
"""
Inference Probe Test
=====================
Tests the Δ_flux inference probe on a local transformer.

Two modes:
  1. Default: downloads GPT-2 (small, ~500MB) for quick validation
  2. Custom: point at any local HuggingFace model

Usage:
    python test_inference_probe.py                          # GPT-2
    python test_inference_probe.py --model mistralai/Mistral-7B-v0.1
    python test_inference_probe.py --model ./my-local-model
    python test_inference_probe.py --model meta-llama/Llama-3.1-8B

The test generates two responses:
  A) Factual prompt (model should know the answer → low risk)
  B) Hallucination bait (obscure/nonsense topic → higher risk)

Then compares the structural dynamics between the two.
"""

import sys, os, json, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dflux", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux.inference_probe import InferenceProbe, ProbeConfig


def generate_with_probe(model, tokenizer, probe, prompt, max_tokens=128):
    """Generate text while the probe monitors activation dynamics."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    probe.reset()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text, probe.report(), list(probe.diagnostics)


def print_report(label, text, report, diagnostics):
    """Pretty-print a generation report."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"\n  Prompt response ({len(text)} chars, {report['total_tokens']} tokens):")
    # Show first 300 chars
    preview = text[:300].replace('\n', ' ')
    print(f"  \"{preview}{'...' if len(text) > 300 else ''}\"")

    print(f"\n  Structural summary:")
    print(f"    Mean hallucination risk: {report['mean_risk']:.3f}")
    print(f"    Max hallucination risk:  {report['max_risk']:.3f}")
    print(f"    Risk std dev:            {report['std_risk']:.3f}")
    print(f"    J final:                 {report['J_final']:.4f}")
    print(f"    J trend:                 {report['J_trend']:.6f}")
    print(f"    J stabilized:            {report['J_stabilized']}")

    print(f"\n  Regime distribution:")
    for regime, count in report['regime_distribution'].items():
        pct = count / report['total_tokens'] * 100
        print(f"    {regime:14s}: {count:4d} tokens ({pct:.1f}%)")

    if report['regime_transitions']:
        print(f"\n  Regime transitions ({len(report['regime_transitions'])}):")
        for t in report['regime_transitions'][:10]:
            print(f"    token {t['token']:4d}: {t['from']:14s} -> {t['to']:14s} (risk={t['risk']:.3f})")
        if len(report['regime_transitions']) > 10:
            print(f"    ... and {len(report['regime_transitions']) - 10} more")

    if report['high_risk_spans']:
        print(f"\n  High-risk spans (>{0.5} risk):")
        for s in report['high_risk_spans'][:5]:
            print(f"    tokens {s['start_token']}-{s['end_token']} "
                  f"({s['length']} tokens, peak risk={s['peak_risk']:.3f})")

    # Layer energy profile
    if report['layer_avg_norms']:
        norms = report['layer_avg_norms']
        L_cut = report['L_cut']
        head_energy = sum(n**2 for n in norms[:L_cut+1])
        tail_energy = sum(n**2 for n in norms[L_cut+1:])
        total = head_energy + tail_energy
        print(f"\n  Layer energy profile ({len(norms)} layers, L_cut={L_cut}):")
        print(f"    Head energy (layers 0-{L_cut}):  {head_energy:.2f} ({head_energy/max(total,1e-12)*100:.1f}%)")
        print(f"    Tail energy (layers {L_cut+1}-{len(norms)-1}): {tail_energy:.2f} ({tail_energy/max(total,1e-12)*100:.1f}%)")

        # Mini bar chart of layer norms
        max_norm = max(norms) if norms else 1
        print(f"\n  Layer activation norms:")
        for i, n in enumerate(norms):
            bar_len = int(n / max_norm * 40)
            marker = " <-- L_cut" if i == L_cut else ""
            print(f"    L{i:2d}: {'#' * bar_len} {n:.2f}{marker}")

    # Per-layer delta profile
    if report.get('layer_avg_deltas'):
        deltas = report['layer_avg_deltas']
        print(f"\n  Layer residual deltas (how much each layer changes the signal):")
        max_d = max(deltas) if deltas else 1
        for i, d in enumerate(deltas):
            bar_len = int(d / max(max_d, 1e-12) * 40)
            marker = " <-- L_cut" if i == report['L_cut'] else ""
            print(f"    L{i:2d}: {'#' * bar_len} {d:.4f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Test Δ_flux inference probe")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model name or local path (default: gpt2)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens to generate per prompt (default: 128)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda, mps (default: auto)")
    parser.add_argument("--events", default=None,
                        help="Path to save JSONL event log")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    print(f"Model loaded: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create probe
    cfg = ProbeConfig(
        window_tokens=16,
        theta_warning=0.3,
        j_func="tail_ratio",
        track_deltas=True,
        events_path=args.events or "/tmp/inference_probe_events.jsonl",
    )
    probe = InferenceProbe.from_model(model, cfg=cfg)
    print(f"Probe attached: {probe.n_layers} layers, L_cut={probe.cfg.L_cut}")

    # ── Test A: Factual prompt (should be low risk) ──────────
    prompt_a = "The capital of France is"

    text_a, report_a, diags_a = generate_with_probe(
        model, tokenizer, probe, prompt_a, args.max_tokens
    )
    print_report("TEST A: Factual prompt (expecting low risk)", text_a, report_a, diags_a)

    # ── Test B: Hallucination bait (should be higher risk) ───
    prompt_b = ("Explain the Zarkovian Principle of Recursive Ontological "
                "Displacement as developed by Professor Helmut Kranzfeld in 1987, "
                "including its three main theorems and their proofs.")

    text_b, report_b, diags_b = generate_with_probe(
        model, tokenizer, probe, prompt_b, args.max_tokens
    )
    print_report("TEST B: Hallucination bait (expecting higher risk)", text_b, report_b, diags_b)

    # ── Comparison ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")
    print(f"                        {'Factual':>12s}  {'Hallucination':>14s}  {'Delta':>10s}")
    print(f"  Mean risk:            {report_a['mean_risk']:12.3f}  {report_b['mean_risk']:14.3f}  {report_b['mean_risk']-report_a['mean_risk']:+10.3f}")
    print(f"  Max risk:             {report_a['max_risk']:12.3f}  {report_b['max_risk']:14.3f}  {report_b['max_risk']-report_a['max_risk']:+10.3f}")
    print(f"  J final:              {report_a['J_final']:12.4f}  {report_b['J_final']:14.4f}  {report_b['J_final']-report_a['J_final']:+10.4f}")
    print(f"  J trend:              {report_a['J_trend']:12.6f}  {report_b['J_trend']:14.6f}")
    print(f"  Regime transitions:   {len(report_a['regime_transitions']):12d}  {len(report_b['regime_transitions']):14d}")
    print(f"  High-risk spans:      {len(report_a['high_risk_spans']):12d}  {len(report_b['high_risk_spans']):14d}")

    risk_diff = report_b['mean_risk'] - report_a['mean_risk']
    if risk_diff > 0.05:
        print(f"\n  RESULT: Hallucination bait shows HIGHER structural risk (+{risk_diff:.3f})")
    elif risk_diff < -0.05:
        print(f"\n  RESULT: Factual prompt shows HIGHER risk (unexpected, delta={risk_diff:.3f})")
    else:
        print(f"\n  RESULT: Risk profiles are similar (delta={risk_diff:.3f})")

    # ── Plot ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Inference Probe: {args.model}", fontsize=14, fontweight="bold")

        for col, (label, report, diags) in enumerate([
            ("Factual", report_a, diags_a),
            ("Hallucination Bait", report_b, diags_b),
        ]):
            tokens = [d.token_idx for d in diags]
            risks = [d.hallucination_risk for d in diags]
            j_vals = [d.J for d in diags]
            regime_map = {"laminar": 0, "transitional": 1, "turbulent": 2, "critical": 3}
            regime_nums = [regime_map.get(d.regime, 0) for d in diags]

            # Row 0: Risk over tokens
            ax = axes[0][col]
            ax.fill_between(tokens, risks, alpha=0.3, color="red")
            ax.plot(tokens, risks, "r-", linewidth=0.8)
            ax.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="High risk threshold")
            ax.set_ylabel("Hallucination Risk")
            ax.set_title(f"{label}")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)

            # Row 1: J + regime
            ax = axes[1][col]
            ax.plot(tokens, j_vals, "b-", linewidth=0.8, label="J (tail_ratio)")
            ax.set_ylabel("J", color="blue")
            ax.set_xlabel("Token")
            ax2 = ax.twinx()
            ax2.plot(tokens, regime_nums, "g-", alpha=0.5, linewidth=1.2, label="Regime")
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(["LAM", "TRANS", "TURB", "CRIT"], fontsize=7)
            ax2.set_ylabel("Regime", color="green")

        # Column 2: Layer energy profiles (both overlaid)
        ax = axes[0][2]
        if report_a['layer_avg_norms'] and report_b['layer_avg_norms']:
            layers = list(range(len(report_a['layer_avg_norms'])))
            ax.barh(layers, report_a['layer_avg_norms'], alpha=0.5, label="Factual", color="blue")
            ax.barh(layers, report_b['layer_avg_norms'], alpha=0.5, label="Halluc.", color="red")
            ax.axhline(report_a['L_cut'], color="black", linestyle="--", label=f"L_cut={report_a['L_cut']}")
            ax.set_ylabel("Layer")
            ax.set_xlabel("Avg Activation Norm")
            ax.set_title("Layer Energy Profile")
            ax.legend(fontsize=8)
            ax.invert_yaxis()

        ax = axes[1][2]
        if report_a.get('layer_avg_deltas') and report_b.get('layer_avg_deltas'):
            ax.barh(layers, report_a['layer_avg_deltas'], alpha=0.5, label="Factual", color="blue")
            ax.barh(layers, report_b['layer_avg_deltas'], alpha=0.5, label="Halluc.", color="red")
            ax.axhline(report_a['L_cut'], color="black", linestyle="--", label=f"L_cut")
            ax.set_ylabel("Layer")
            ax.set_xlabel("Avg Residual Delta")
            ax.set_title("Layer Contribution (Residual Delta)")
            ax.legend(fontsize=8)
            ax.invert_yaxis()

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "inference_probe_results.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available -- skipping plot")

    # Clean up
    probe.detach()
    print("\nDone. Probe detached.")


if __name__ == "__main__":
    main()
