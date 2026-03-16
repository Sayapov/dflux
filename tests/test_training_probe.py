#!/usr/bin/env python3
"""
Training + Inference Probe: Watch a model learn under the Δ_flux microscope.
===========================================================================

Fine-tunes GPT-2 on wikitext-2 while:
  1. AXE-NS monitors gradient dynamics (training stability)
  2. Inference probe monitors activation dynamics (generation quality)

Every N steps, pauses training and runs the probe on two prompts:
  A) Factual: "The capital of France is"
  B) Hallucination bait: made-up topic

You get to watch:
  - How the model's internal dynamics evolve as it learns
  - Whether hallucination risk decreases with training
  - Whether regime transitions in training correlate with
    generation quality changes
  - The J trajectory during both training and inference

Designed for Mac Studio Ultra (96GB, MPS backend).
Also works on CUDA or CPU (slower).

Usage:
    python test_training_probe.py                    # defaults
    python test_training_probe.py --steps 1000       # more training
    python test_training_probe.py --device cpu        # force CPU
    python test_training_probe.py --model gpt2-medium # bigger model
"""

import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dflux", "src"))

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from dflux.axe_ns import AXEEngine, Regime
from dflux.inference_probe import InferenceProbe, ProbeConfig


def get_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def probe_generation(model, tokenizer, probe, prompt, max_tokens=64):
    """Run inference probe on a single prompt."""
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
    parser = argparse.ArgumentParser(description="Training + Inference Probe")
    parser.add_argument("--model", default="gpt2", help="HF model (default: gpt2)")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--probe-every", type=int, default=50, help="Probe interval")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--inject-chaos", action="store_true",
                        help="Inject instability at 60%% of training")
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("Training + Inference Probe")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Device:      {device}")
    print(f"Steps:       {args.steps}")
    print(f"Probe every: {args.probe_every} steps")
    print(f"LR:          {args.lr}")
    print(f"Chaos:       {'YES' if args.inject_chaos else 'no'}")
    print("=" * 70)

    # ── Load model & tokenizer ───────────────────────────────
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use float32 on MPS — plenty of RAM and avoids NaN during generation
    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {type(model).__name__}, {n_params:,} params")

    # ── Load dataset ─────────────────────────────────────────
    print("Loading wikitext-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    tokenized = dataset.filter(lambda x: len(x["text"]) > 50).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Simple dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    data_iter = iter(dataloader)

    def get_batch():
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return {k: v.to(device) for k, v in batch.items()}

    # ── Setup optimizer with per-layer groups ────────────────
    # Find transformer layers for per-layer param groups
    layer_modules = InferenceProbe._find_transformer_layers(model)
    n_layers = len(layer_modules)

    # Create param groups: embedding + each transformer layer + LM head
    param_groups = []

    # Embedding params
    embed_params = []
    for name, param in model.named_parameters():
        is_layer_param = False
        for i, layer in enumerate(layer_modules):
            for lname, _ in layer.named_parameters():
                if name.endswith(lname):
                    is_layer_param = True
                    break
            if is_layer_param:
                break
        if not is_layer_param:
            embed_params.append(param)

    if embed_params:
        param_groups.append({"params": embed_params, "lr": args.lr, "name": "embed+head"})

    # Per-layer params
    for i, layer in enumerate(layer_modules):
        param_groups.append({
            "params": list(layer.parameters()),
            "lr": args.lr,
            "name": f"layer_{i}",
        })

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.01)
    print(f"Optimizer: {len(param_groups)} param groups ({n_layers} transformer layers)")

    # ── Setup AXE-NS engine ──────────────────────────────────
    engine = AXEEngine.from_optimizer(
        optimizer,
        window_steps=50,
        theta_warning=0,           # Auto-calibrate from warmup
        beta=100,                  # Check every 100 steps
        csv_path=os.path.join(output_dir, "training_probe_axe.csv"),
        events_path=os.path.join(output_dir, "training_probe_events.jsonl"),
        run_id="training-probe",
        model_id=args.model,
        j_func="tail_ratio",
        stagnation_eps=1e-3,
        warmup_steps=50,           # Observe 50 steps before acting
        ema_alpha=0.05,            # Smooth J heavily for transformers
        monotone_tolerance=0.05,   # 5% tolerance on smoothed J
        violations_required=5,     # Need 5 consecutive before RENORMALIZE
    )
    print(f"AXE-NS: L_cut={engine.cfg.L_cut}, J_func=tail_ratio, warmup={engine.cfg.warmup_steps}")

    # ── Setup inference probe ────────────────────────────────
    probe_cfg = ProbeConfig(
        window_tokens=16,
        j_func="tail_ratio",
        track_deltas=True,
    )
    inference_probe = InferenceProbe.from_model(model, cfg=probe_cfg)
    print(f"Inference probe: {inference_probe.n_layers} layers, L_cut={inference_probe.cfg.L_cut}")

    # ── Prompts for periodic probing ─────────────────────────
    FACTUAL_PROMPT = "The capital of France is"
    HALLUC_PROMPT = ("Explain the Zarkovian Principle of Recursive Ontological "
                     "Displacement, including its three main theorems.")

    # ── Training loop ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}\n")

    # Tracking
    training_losses = []
    training_regimes = []
    training_J = []
    training_actions = []
    probe_snapshots = []  # (step, factual_report, halluc_report, factual_text, halluc_text)

    chaos_step = int(args.steps * 0.6) if args.inject_chaos else None
    start_time = time.time()

    for step in range(args.steps):
        # ── Get batch ──
        batch = get_batch()

        # ── Forward + backward ──
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        # ── AXE-NS step ──
        current_lr = optimizer.param_groups[0]["lr"]
        action = engine.step(step, loss=loss.item(), lr=current_lr)

        # Apply engine recommendations
        if action.kind == "extract":
            engine.apply_extract(optimizer)
            training_actions.append(("extract", step, action.reason))
            print(f"  [step {step}] EXTRACT: {action.reason}")
        elif action.kind == "renormalize":
            engine.apply_renormalize(optimizer)
            training_actions.append(("renormalize", step, action.reason))
            print(f"  [step {step}] RENORMALIZE: {action.reason}")

        optimizer.step()

        # Track
        training_losses.append(loss.item())
        training_regimes.append(engine.regime.value)
        training_J.append(engine.J.current)

        # ── Inject chaos ──
        if args.inject_chaos and step == chaos_step:
            print(f"\n  {'!'*50}")
            print(f"  STEP {step}: INJECTING CHAOS (3x LR on deep layers)")
            print(f"  {'!'*50}\n")
            for i, pg in enumerate(optimizer.param_groups):
                if i > engine.cfg.L_cut:
                    pg["lr"] = pg["lr"] * 3.0

        # ── Periodic logging ──
        if step % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / max(elapsed, 1)
            print(f"  step {step:4d} | loss {loss.item():.4f} | J {engine.J.current:.4f} "
                  f"| regime {engine.regime.value:12s} | depth {engine.depth} "
                  f"| {steps_per_sec:.1f} steps/s")

        # ── Periodic inference probe ──
        if step % args.probe_every == 0 or step == args.steps - 1:
            model.eval()

            # Factual probe
            f_text, f_report = probe_generation(
                model, tokenizer, inference_probe, FACTUAL_PROMPT, max_tokens=48
            )

            # Hallucination probe
            h_text, h_report = probe_generation(
                model, tokenizer, inference_probe, HALLUC_PROMPT, max_tokens=48
            )

            probe_snapshots.append({
                "step": step,
                "loss": loss.item(),
                "training_regime": engine.regime.value,
                "training_J": engine.J.current,
                "factual": {
                    "text": f_text[:200],
                    "mean_risk": f_report.get("mean_risk", 0),
                    "max_risk": f_report.get("max_risk", 0),
                    "J_final": f_report.get("J_final", 0),
                    "J_trend": f_report.get("J_trend", 0),
                    "regimes": f_report.get("regime_distribution", {}),
                },
                "hallucination": {
                    "text": h_text[:200],
                    "mean_risk": h_report.get("mean_risk", 0),
                    "max_risk": h_report.get("max_risk", 0),
                    "J_final": h_report.get("J_final", 0),
                    "J_trend": h_report.get("J_trend", 0),
                    "regimes": h_report.get("regime_distribution", {}),
                },
            })

            print(f"\n  --- Probe @ step {step} ---")
            print(f"  Factual:   risk={f_report.get('mean_risk',0):.3f}, "
                  f"J={f_report.get('J_final',0):.4f}, "
                  f"trend={f_report.get('J_trend',0):.6f}")
            print(f"  Halluc:    risk={h_report.get('mean_risk',0):.3f}, "
                  f"J={h_report.get('J_final',0):.4f}, "
                  f"trend={h_report.get('J_trend',0):.6f}")
            risk_gap = (h_report.get('mean_risk',0) - f_report.get('mean_risk',0))
            print(f"  Risk gap:  {risk_gap:+.3f} "
                  f"({'halluc higher' if risk_gap > 0 else 'factual higher'})")
            print()

            model.train()

    # ── Close engine ──
    engine.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete: {args.steps} steps in {elapsed:.1f}s "
          f"({args.steps/elapsed:.1f} steps/s)")

    # ── Engine summary ───────────────────────────────────────
    summary = engine.summary()
    print(f"\n{'=' * 70}")
    print("AXE-NS ENGINE SUMMARY")
    print(f"{'=' * 70}")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")

    # ── Save probe snapshots ─────────────────────────────────
    snapshots_path = os.path.join(output_dir, "training_probe_snapshots.json")
    with open(snapshots_path, "w") as f:
        json.dump(probe_snapshots, f, indent=2)
    print(f"\nProbe snapshots saved to: {snapshots_path}")

    # ── Generate plots ───────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(18, 16))
        gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)
        fig.suptitle(f"Training + Inference Probe: {args.model}", fontsize=14, fontweight="bold")

        steps_range = list(range(len(training_losses)))

        # ── Panel 1: Training loss ──
        ax = fig.add_subplot(gs[0, 0])
        ax.semilogy(steps_range, training_losses, "k-", alpha=0.7, linewidth=0.6)
        if chaos_step:
            ax.axvline(chaos_step, color="red", linestyle="--", alpha=0.5, label="Chaos")
        for a in training_actions:
            color = "blue" if a[0] == "extract" else "red"
            ax.axvline(a[1], color=color, alpha=0.3, linewidth=0.5)
        ax.set_ylabel("Loss (log)")
        ax.set_title("Training Loss")
        ax.legend(fontsize=8)

        # ── Panel 2: Training J ──
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(steps_range, training_J, "b-", linewidth=0.6)
        if chaos_step:
            ax.axvline(chaos_step, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("J (tail_ratio)")
        ax.set_title("Training J (Complexity Functional)")

        # ── Panel 3: Training regime ──
        ax = fig.add_subplot(gs[1, 0])
        regime_map = {"laminar": 0, "transitional": 1, "turbulent": 2, "critical": 3}
        regime_nums = [regime_map.get(r, 0) for r in training_regimes]
        ax.fill_between(steps_range, regime_nums, alpha=0.3, color="green")
        ax.plot(steps_range, regime_nums, "g-", linewidth=0.8)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Laminar", "Transitional", "Turbulent", "Critical"])
        ax.set_ylabel("Regime")
        ax.set_title("Training Regime")

        # ── Panel 4: Inference risk evolution ──
        ax = fig.add_subplot(gs[1, 1])
        probe_steps = [s["step"] for s in probe_snapshots]
        fact_risks = [s["factual"]["mean_risk"] for s in probe_snapshots]
        hall_risks = [s["hallucination"]["mean_risk"] for s in probe_snapshots]
        ax.plot(probe_steps, fact_risks, "b-o", markersize=4, label="Factual")
        ax.plot(probe_steps, hall_risks, "r-o", markersize=4, label="Hallucination")
        ax.fill_between(probe_steps, fact_risks, hall_risks, alpha=0.15, color="orange")
        if chaos_step:
            ax.axvline(chaos_step, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Mean Risk")
        ax.set_title("Inference Risk Evolution During Training")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

        # ── Panel 5: Inference J evolution ──
        ax = fig.add_subplot(gs[2, 0])
        fact_J = [s["factual"]["J_final"] for s in probe_snapshots]
        hall_J = [s["hallucination"]["J_final"] for s in probe_snapshots]
        ax.plot(probe_steps, fact_J, "b-o", markersize=4, label="Factual J")
        ax.plot(probe_steps, hall_J, "r-o", markersize=4, label="Hallucination J")
        if chaos_step:
            ax.axvline(chaos_step, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("J (tail_ratio)")
        ax.set_title("Inference J Evolution")
        ax.legend(fontsize=8)

        # ── Panel 6: Risk gap ──
        ax = fig.add_subplot(gs[2, 1])
        risk_gaps = [h - f for h, f in zip(hall_risks, fact_risks)]
        colors = ["red" if g > 0 else "blue" for g in risk_gaps]
        ax.bar(probe_steps, risk_gaps, width=max(1, args.probe_every * 0.8),
               color=colors, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Risk Gap (halluc - factual)")
        ax.set_title("Hallucination Risk Separation")
        ax.set_xlabel("Step")

        # ── Panel 7: J trend comparison ──
        ax = fig.add_subplot(gs[3, 0])
        fact_trend = [s["factual"]["J_trend"] for s in probe_snapshots]
        hall_trend = [s["hallucination"]["J_trend"] for s in probe_snapshots]
        ax.plot(probe_steps, fact_trend, "b-o", markersize=4, label="Factual J_trend")
        ax.plot(probe_steps, hall_trend, "r-o", markersize=4, label="Hallucination J_trend")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("J trend")
        ax.set_xlabel("Step")
        ax.set_title("Inference J Trend (+ = degrading, - = compressing)")
        ax.legend(fontsize=8)

        # ── Panel 8: Training actions timeline ──
        ax = fig.add_subplot(gs[3, 1])
        extract_steps = [a[1] for a in training_actions if a[0] == "extract"]
        renorm_steps = [a[1] for a in training_actions if a[0] == "renormalize"]
        ax.eventplot([extract_steps], lineoffsets=1, linelengths=0.5, colors="blue", label="EXTRACT")
        ax.eventplot([renorm_steps], lineoffsets=2, linelengths=0.5, colors="red", label="RENORMALIZE")
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["EXTRACT", "RENORMALIZE"])
        ax.set_xlabel("Step")
        ax.set_title("AXE-NS Actions Timeline")
        ax.set_xlim(0, args.steps)

        plot_path = os.path.join(output_dir, "training_probe_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available -- skipping plot")

    # Clean up
    inference_probe.detach()
    print("\nDone.")


if __name__ == "__main__":
    main()
