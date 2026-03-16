#!/usr/bin/env python3
"""
Δ_flux real-world test — GPT-2 fine-tune on Mac Studio Ultra.

What this does:
  1. Loads GPT-2 (124M params) from HuggingFace
  2. Fine-tunes on wikitext-2 for 500 steps (stable)
  3. At step 500: corrupts 30% of training data (token shuffling)
     AND bumps LR 3x — simulating a subtle pipeline bug
  4. Continues for 500 more steps
  5. Δ_flux watches the whole time

Why this test matters:
  Transformers have attention heads that can degrade independently
  of the output loss. Skip connections mask deep-layer instability.
  This is where Δ_flux should detect structural problems BEFORE
  the perplexity/loss shows them.

Requirements:
  pip3 install torch transformers datasets dflux

Run:
  python3 dflux_transformer_test.py
"""

import os
import sys
import json
import math
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset

# ── Check for dflux ──
try:
    from dflux.adapters.hf_trainer import DFluxCallback
except ImportError:
    print("dflux not found. Install with: pip3 install dflux")
    print("Or if running from source, add the dflux/src dir to PYTHONPATH")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

MODEL_NAME = "gpt2"                  # 124M params — fits easily in 96GB
STABLE_STEPS = 500                   # clean training
CORRUPT_STEPS = 500                  # training with corrupted data
TOTAL_STEPS = STABLE_STEPS + CORRUPT_STEPS
BATCH_SIZE = 8
BLOCK_SIZE = 128                     # context length for training
LEARNING_RATE = 5e-5
CORRUPT_FRACTION = 0.3               # 30% of tokens get shuffled
LR_SPIKE_FACTOR = 3                  # 3x LR bump at corruption point
OUTPUT_DIR = "./dflux_gpt2_test"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Stable steps: {STABLE_STEPS}, Corrupt steps: {CORRUPT_STEPS}")
print()


# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════

print("Loading tokenizer and dataset...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Tokenize and chunk into blocks
def tokenize_and_chunk(examples):
    tokens = tokenizer(examples["text"], truncation=False)["input_ids"]
    # Flatten all tokens
    all_tokens = [t for seq in tokens for t in seq]
    # Chunk into blocks
    chunks = []
    for i in range(0, len(all_tokens) - BLOCK_SIZE, BLOCK_SIZE):
        chunks.append(all_tokens[i:i+BLOCK_SIZE])
    return {"input_ids": chunks}

tokenized = dataset.map(tokenize_and_chunk, batched=True, remove_columns=["text"],
                         batch_size=1000)
print(f"Training chunks: {len(tokenized)}")


# Custom dataset that can inject corruption mid-training
class CorruptableDataset(Dataset):
    def __init__(self, data, corrupt_fraction=0.0):
        self.data = data
        self.corrupt_fraction = corrupt_fraction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)
        labels = input_ids.clone()

        # Corrupt: shuffle random tokens within the sequence
        if self.corrupt_fraction > 0:
            n_corrupt = int(len(input_ids) * self.corrupt_fraction)
            corrupt_idx = torch.randperm(len(input_ids))[:n_corrupt]
            shuffled = input_ids[corrupt_idx][torch.randperm(n_corrupt)]
            input_ids[corrupt_idx] = shuffled
            # Labels stay clean — model gets conflicting signal
            # (corrupted input, clean target)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}

train_dataset = CorruptableDataset(tokenized)


# ═══════════════════════════════════════════════════════════
# CORRUPTION INJECTION CALLBACK
# ═══════════════════════════════════════════════════════════

class CorruptionInjector(TrainerCallback):
    """At the specified step, corrupt the dataset and spike the LR."""

    def __init__(self, inject_at_step, corrupt_fraction, lr_spike_factor):
        self.inject_at_step = inject_at_step
        self.corrupt_fraction = corrupt_fraction
        self.lr_spike_factor = lr_spike_factor
        self.injected = False

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step >= self.inject_at_step and not self.injected:
            print(f"\n>>> INJECTING INSTABILITY at step {state.global_step}")
            print(f"    Corrupting {self.corrupt_fraction:.0%} of tokens")
            print(f"    Spiking LR by {self.lr_spike_factor}x\n")

            # Corrupt the dataset
            trainer = kwargs.get("model", None)
            # Access the train dataset through the trainer
            if hasattr(state, "_trainer"):
                state._trainer.train_dataset.corrupt_fraction = self.corrupt_fraction

            # Spike LR
            optimizer = kwargs.get("optimizer")
            if optimizer:
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] * self.lr_spike_factor

            self.injected = True

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # Also inject via optimizer access since on_step_begin might not have optimizer
        if state.global_step >= self.inject_at_step and not self.injected:
            optimizer = kwargs.get("optimizer")
            if optimizer:
                print(f"\n>>> INJECTING INSTABILITY at step {state.global_step}")
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["lr"] * self.lr_spike_factor
                self.injected = True


# ═══════════════════════════════════════════════════════════
# LOGGING CALLBACK
# ═══════════════════════════════════════════════════════════

class MetricsLogger(TrainerCallback):
    """Log loss at regular intervals and track for analysis."""

    def __init__(self):
        self.losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)
            step = state.global_step
            if step % 50 == 0:
                print(f"  step {step:4d}: loss={logs['loss']:.4f}")


# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════

print(f"\nLoading {MODEL_NAME}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=TOTAL_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=999999,  # don't save checkpoints
    warmup_steps=50,
    lr_scheduler_type="cosine",
    report_to="none",
    use_mps_device=(DEVICE == "mps"),
    dataloader_num_workers=0,
)

metrics_logger = MetricsLogger()

dflux_callback = DFluxCallback(
    run_id="gpt2-instability-test",
    model_id="gpt2-124M",
    L_cut="auto",
    window_steps=100,
    theta_warning=0.5,      # will need calibration — start sensitive
    out_dir=os.path.join(OUTPUT_DIR, "dflux_logs"),
)

corruption_injector = CorruptionInjector(
    inject_at_step=STABLE_STEPS,
    corrupt_fraction=CORRUPT_FRACTION,
    lr_spike_factor=LR_SPIKE_FACTOR,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[dflux_callback, corruption_injector, metrics_logger],
)

# Give the corruption injector access to the dataset
# (Hack: set corrupt_fraction on dataset directly when triggered)
original_get_train_dataloader = trainer.get_train_dataloader

def patched_get_train_dataloader():
    dl = original_get_train_dataloader()
    return dl

trainer.get_train_dataloader = patched_get_train_dataloader

print(f"\nStarting training...")
print(f"  Steps 0-{STABLE_STEPS}: Clean training")
print(f"  Steps {STABLE_STEPS}-{TOTAL_STEPS}: Corrupted data + LR spike")
print()

trainer.train()

# ═══════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Read Δ_flux events
events_path = os.path.join(OUTPUT_DIR, "dflux_logs", "dflux_events.jsonl")
dflux_events = []
if os.path.exists(events_path):
    with open(events_path) as f:
        dflux_events = [json.loads(l) for l in f if l.strip()]

print(f"Total Δ_flux events: {len(dflux_events)}")

pre_inject = [e for e in dflux_events if e["t_end"] < STABLE_STEPS]
post_inject = [e for e in dflux_events if e["t_start"] >= STABLE_STEPS]
print(f"  Before injection: {len(pre_inject)}")
print(f"  After injection:  {len(post_inject)}")

if dflux_events:
    first = dflux_events[0]
    print(f"\nFirst event: steps {first['t_start']}-{first['t_end']}, "
          f"Δ_flux={first['delta_flux']:.4f}, severity={first['severity']}")

# Read CSV for plotting
csv_path = os.path.join(OUTPUT_DIR, "dflux_logs", "dflux.csv")
if os.path.exists(csv_path):
    import csv as csv_mod
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    print(f"\nCSV rows: {len(rows)}")

    if rows:
        # Plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [int(r["step"]) for r in rows]
            csv_losses = [float(r["loss"]) if r["loss"] else None for r in rows]
            etails = [float(r["E_tail"]) for r in rows]
            dfluxes = [float(r["delta_flux_window"]) for r in rows]

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            fig.patch.set_facecolor("#0a0a0f")
            for ax in axes:
                ax.set_facecolor("#12121a")
                ax.tick_params(colors="#6b6b80")
                for s in ax.spines.values(): s.set_color("#1e1e2e")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.axvspan(STABLE_STEPS, TOTAL_STEPS, alpha=0.06, color="#ff3b5c")
                ax.axvline(x=STABLE_STEPS, color="#ff3b5c", linestyle="--", alpha=0.5)

            # Loss
            valid_losses = [(s, l) for s, l in zip(steps, csv_losses) if l is not None]
            if valid_losses:
                sl, ll = zip(*valid_losses)
                axes[0].plot(sl, ll, color="#5ce0d8", linewidth=0.5)
                axes[0].set_ylabel("Loss", color="#e0e0e8")
                axes[0].set_title("Δ_flux: GPT-2 Fine-tune Instability Test (Mac Studio Ultra)",
                                  color="#e0e0e8", fontsize=13, fontweight="bold")

            # Mark events
            for e in dflux_events:
                for ax in axes:
                    ax.axvline(x=e["t_end"], color="#f1c40f", alpha=0.3, linewidth=1)

            axes[1].plot(steps, etails, color="#7b5cff", linewidth=0.5)
            axes[1].set_ylabel("E_tail", color="#e0e0e8")

            axes[2].plot(steps, dfluxes, color="#2ecc71", linewidth=0.5)
            axes[2].set_ylabel("Δ_flux", color="#e0e0e8")
            axes[2].set_xlabel("Step", color="#e0e0e8")

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_DIR, "dflux_gpt2_results.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"\nPlot saved: {plot_path}")
        except ImportError:
            print("matplotlib not installed — skipping plot")

print(f"\nFull results in: {OUTPUT_DIR}/dflux_logs/")
print("  dflux.csv — per-step metrics")
print("  dflux_events.jsonl — stability events")
