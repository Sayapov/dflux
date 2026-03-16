#!/usr/bin/env python3
"""
AXE-NS Integration Test
========================
Trains a small MLP, injects instability, and watches the engine:
  1. Detect regime transition (laminar → turbulent)
  2. Recommend EXTRACT with specific LR adjustments
  3. Apply the fix → J decreases, regime recovers
  4. If things get worse → RENORMALIZE fires

No GPU needed — runs in ~30 seconds on CPU.
"""

import sys, os, json
import numpy as np

# Add dflux src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dflux", "src"))

import torch
import torch.nn as nn
from dflux.axe_ns import AXEEngine, AXEConfig, Regime

# ── Reproducibility ──────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Synthetic data ───────────────────────────────────────
N, D_in, D_hid, D_out = 512, 20, 64, 1

X = torch.randn(N, D_in)
W_true = torch.randn(D_in, D_out) * 0.3
y_clean = X @ W_true + torch.randn(N, D_out) * 0.1
y = y_clean.clone()  # Mutable copy, separate from graph

# ── Model: 6-layer MLP (6 param groups) ─────────────────
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(D_in, D_hid),  nn.ReLU(),
            nn.Linear(D_hid, D_hid), nn.ReLU(),
            nn.Linear(D_hid, D_hid), nn.ReLU(),
            nn.Linear(D_hid, D_hid), nn.ReLU(),
            nn.Linear(D_hid, D_hid), nn.ReLU(),
            nn.Linear(D_hid, D_out),
        )

    def forward(self, x):
        return self.layers(x)

model = DeepMLP()
criterion = nn.MSELoss()

# Give each linear layer its own param group
param_groups = []
for module in model.layers:
    if isinstance(module, nn.Linear):
        param_groups.append({"params": list(module.parameters()), "lr": 1e-3})

optimizer = torch.optim.Adam(param_groups, lr=1e-3, weight_decay=1e-4)

# ── Create AXE-NS engine ────────────────────────────────
# Use tail_ratio for J so we measure structural imbalance, not just raw energy.
# Higher stagnation_eps prevents false EXTRACT triggers during normal convergence.
engine = AXEEngine.from_optimizer(
    optimizer,
    window_steps=30,
    theta_warning=0.5,
    beta=40,
    csv_path="/tmp/axe_ns_test.csv",
    events_path="/tmp/axe_ns_test_events.jsonl",
    run_id="integration-test",
    model_id="deep-mlp-6layer",
    j_func="tail_ratio",
    stagnation_eps=1e-3,
    monotone_tolerance=0.02,
)

print("=" * 70)
print("AXE-NS Integration Test")
print("=" * 70)
print(f"Model: 6-layer MLP ({sum(p.numel() for p in model.parameters()):,} params)")
print(f"Param groups: {len(param_groups)}, L_cut: {engine.cfg.L_cut}")
print(f"J functional: {engine.cfg.j_func}")
print(f"Plan: 200 stable → inject chaos → let engine fix it → 400 more")
print("=" * 70)

# ── Training loop ────────────────────────────────────────
total_steps = 600
inject_step = 200
chaos_injected = False
actions_taken = []
losses = []
regimes = []
J_values = []
lrs = []

for step in range(total_steps):
    # ── Inject instability once at step 200 ──
    if step == inject_step and not chaos_injected:
        chaos_injected = True
        print(f"\n{'!'*60}")
        print(f"  STEP {step}: INJECTING INSTABILITY")
        print(f"  → Corrupting 40% of labels + 5x LR spike on deep layers")
        print(f"{'!'*60}\n")

        # Corrupt labels (create new tensor, no in-place on graph)
        mask = torch.rand(N) < 0.4
        y = y_clean.clone()
        y[mask] = torch.randn(mask.sum(), D_out) * 5.0

        # Spike LR on tail groups (deep layers)
        for i, pg in enumerate(optimizer.param_groups):
            if i > engine.cfg.L_cut:
                pg["lr"] = pg["lr"] * 5.0

    # ── Forward pass (detach targets from graph) ──
    pred = model(X)
    loss = criterion(pred, y.detach())

    # ── Backward ──
    optimizer.zero_grad()
    loss.backward()

    # ── AXE-NS step ──
    current_lr = optimizer.param_groups[0]["lr"]
    action = engine.step(step, loss=loss.item(), lr=current_lr)

    # ── Apply engine recommendations ──
    if action.kind == "extract":
        changes = engine.apply_extract(optimizer)
        actions_taken.append(("extract", step, action.reason))
        print(f"  EXTRACT @ step {step}: {action.reason}")
        for k, v in changes.items():
            if "lr" in k:
                print(f"      {k}: {v['old']:.2e} -> {v['new']:.2e}")

    elif action.kind == "renormalize":
        changes = engine.apply_renormalize(optimizer)
        actions_taken.append(("renormalize", step, action.reason))
        print(f"  RENORMALIZE @ step {step}: {action.reason}")
        for k, v in changes.items():
            if "lr" in k:
                print(f"      {k}: {v['old']:.2e} -> {v['new']:.2e}")

    elif action.kind == "alert":
        actions_taken.append(("alert", step, action.reason))
        if step % 50 == 0 or step == inject_step:
            print(f"  ALERT @ step {step}: {action.reason}")

    elif action.kind == "stabilized":
        if not any(a[0] == "stabilized" and a[1] > step - 20 for a in actions_taken):
            actions_taken.append(("stabilized", step, action.reason))
            print(f"  STABILIZED @ step {step}: {action.reason}")

    # ── Optimizer step ──
    optimizer.step()

    # ── Track ──
    losses.append(loss.item())
    regimes.append(engine.regime.value)
    J_values.append(engine.J.current)
    lrs.append(current_lr)

    # ── Log every 50 steps ──
    if step % 50 == 0:
        print(f"  step {step:4d} | loss {loss.item():8.4f} | J {engine.J.current:.4f} "
              f"| regime {engine.regime.value:14s} | depth {engine.depth} "
              f"| lr {current_lr:.2e}")

# ── Close engine ──
engine.close()

# ── Summary ──────────────────────────────────────────────
summary = engine.summary()
print(f"\n{'=' * 70}")
print("ENGINE SUMMARY")
print(f"{'=' * 70}")
for k, v in summary.items():
    print(f"  {k:25s}: {v}")

# ── Read back events ─────────────────────────────────────
print(f"\n{'=' * 70}")
print("EVENT LOG")
print(f"{'=' * 70}")
with open("/tmp/axe_ns_test_events.jsonl") as f:
    events = [json.loads(line) for line in f if line.strip()]

for ev in events[:20]:
    print(f"  step {ev['step']:4d} | {ev['action']:12s} | regime {ev['regime']:14s} "
          f"| J {ev['J']:.4f} | depth {ev['depth']}")

if len(events) > 20:
    print(f"  ... and {len(events) - 20} more events")
print(f"\nTotal events: {len(events)}")

# ── Assertions ───────────────────────────────────────────
print(f"\n{'=' * 70}")
print("VERIFICATION")
print(f"{'=' * 70}")

extract_count = sum(1 for a in actions_taken if a[0] == "extract")
renorm_count = sum(1 for a in actions_taken if a[0] == "renormalize")
alert_count = sum(1 for a in actions_taken if a[0] == "alert")
checks = []

# Check 1: Engine detected instability
detected = extract_count > 0 or renorm_count > 0 or alert_count > 0
checks.append(("Engine detected instability", detected))
print(f"  {'PASS' if detected else 'FAIL'} Engine detected instability "
      f"({extract_count} extracts, {renorm_count} renorms, {alert_count} alerts)")

# Check 2: Detection near injection
action_steps_after_inject = [a[1] for a in actions_taken
                             if a[0] in ("extract", "renormalize", "alert") and a[1] >= inject_step]
if action_steps_after_inject:
    first = min(action_steps_after_inject)
    near = first <= inject_step + 80
    checks.append(("Detection near injection point", near))
    print(f"  {'PASS' if near else 'FAIL'} First post-injection action at step {first} "
          f"(injected at {inject_step})")
else:
    checks.append(("Detection near injection point", False))
    print(f"  FAIL No actions after injection!")

# Check 3: Regime transition
saw_non_laminar = any(r != "laminar" for r in regimes[inject_step:])
checks.append(("Regime transition detected", saw_non_laminar))
print(f"  {'PASS' if saw_non_laminar else 'FAIL'} Saw non-laminar regime after injection")

# Check 4: Witnesses
has_witnesses = summary["witness_count"] > 0
checks.append(("Witness records created", has_witnesses))
print(f"  {'PASS' if has_witnesses else 'FAIL'} {summary['witness_count']} witness records")

# Check 5: J recovered
if len(J_values) > inject_step + 100:
    j_peak = max(J_values[inject_step:inject_step+100])
    j_end = J_values[-1]
    recovered = j_end < j_peak
    checks.append(("J recovered after intervention", recovered))
    print(f"  {'PASS' if recovered else 'FAIL'} J peak: {j_peak:.4f}, J final: {j_end:.4f}")

all_passed = all(c[1] for c in checks)
print(f"\n  {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
print(f"{'=' * 70}")

# ── Generate plot ────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("AXE-NS Integration Test: Detect -> Diagnose -> Fix", fontsize=14, fontweight="bold")

    steps = list(range(total_steps))

    # Panel 1: Loss
    ax = axes[0]
    ax.semilogy(steps, losses, "k-", alpha=0.7, linewidth=0.8)
    ax.axvspan(inject_step, inject_step+50, alpha=0.3, color="red", label="Chaos injection")
    for a in actions_taken:
        if a[0] == "extract":
            ax.axvline(a[1], color="blue", alpha=0.4, linestyle="--", linewidth=0.7)
        elif a[0] == "renormalize":
            ax.axvline(a[1], color="red", alpha=0.5, linestyle="-", linewidth=1.0)
    ax.set_ylabel("Loss (log)")
    ax.legend(loc="upper right")
    ax.set_title("Training Loss")

    # Panel 2: J functional
    ax = axes[1]
    ax.plot(steps, J_values, "b-", linewidth=0.8)
    ax.axvspan(inject_step, inject_step+50, alpha=0.3, color="red")
    for a in actions_taken:
        if a[0] == "extract":
            ax.axvline(a[1], color="blue", alpha=0.4, linestyle="--", linewidth=0.7)
        elif a[0] == "renormalize":
            ax.axvline(a[1], color="red", alpha=0.5, linestyle="-", linewidth=1.0)
    ax.set_ylabel("J (tail_ratio)")
    ax.set_title("Monotone Complexity Functional J")

    # Panel 3: Regime
    regime_map = {"laminar": 0, "transitional": 1, "turbulent": 2, "critical": 3}
    regime_nums = [regime_map.get(r, 0) for r in regimes]
    ax = axes[2]
    ax.fill_between(steps, regime_nums, alpha=0.4, color="green")
    ax.plot(steps, regime_nums, "g-", linewidth=1.2)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Laminar", "Transitional", "Turbulent", "Critical"])
    ax.axvspan(inject_step, inject_step+50, alpha=0.3, color="red")
    ax.set_ylabel("Regime")
    ax.set_title("Flow Regime Classification")

    # Panel 4: Learning rate
    ax = axes[3]
    ax.semilogy(steps, lrs, "m-", linewidth=0.8)
    ax.axvspan(inject_step, inject_step+50, alpha=0.3, color="red")
    for a in actions_taken:
        if a[0] == "extract":
            ax.axvline(a[1], color="blue", alpha=0.4, linestyle="--", linewidth=0.7)
        elif a[0] == "renormalize":
            ax.axvline(a[1], color="red", alpha=0.5, linestyle="-", linewidth=1.0)
    ax.set_ylabel("LR (log)")
    ax.set_xlabel("Step")
    ax.set_title("Learning Rate (adjusted by engine)")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="blue", linestyle="--", label="EXTRACT"),
        Line2D([0], [0], color="red", linestyle="-", label="RENORMALIZE"),
        plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.3, label="Chaos window"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plot_path = os.path.join(os.path.dirname(__file__), "axe_ns_test_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

except ImportError:
    print("\nmatplotlib not available -- skipping plot generation")
