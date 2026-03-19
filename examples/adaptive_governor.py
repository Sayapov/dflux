#!/usr/bin/env python3
"""
Adaptive Governor Demo — windowed, EMA-tracked, multi-signal scaling.
=====================================================================

Runs the adaptive governor on Qwen3.5-9B with configurable modes:

  --mode compass     Use dilution_survival 0.8 profile as target (recommended)
  --mode signal      Pure signal-driven, no target profile
  --mode chat        Interactive chat with adaptive governor active

Usage:
    python examples/adaptive_governor.py --device mps --dtype bfloat16
    python examples/adaptive_governor.py --mode signal --window 16
    python examples/adaptive_governor.py --mode chat --device mps
"""

import sys, os, json, argparse, time, gc

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoTokenizer, AutoConfig

from dflux.adaptive_governor import AdaptiveGovernor, AdaptiveConfig


def load_model(model_name, device, dtype):
    """Load model (handles Qwen3.5 key remapping)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_qwen35 = getattr(config, "model_type", "") == "qwen3_5"

    if is_qwen35:
        # Import the proven loader from head_ablation
        sys.path.insert(0, os.path.dirname(__file__))
        from head_ablation import load_model as _load
        return _load(model_name, device, dtype)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True, device_map=device
        )
        model.eval()
        return model


def generate(model, tokenizer, prompt, device, max_tokens=256):
    """Generate with greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


PROMPTS = [
    "What is the temperature on the far side of the moon, and what would happen to a cucumber if it suddenly appeared there.",
    "Write a short poem about a robot who discovers it can dream.",
    "Solve step by step: If a train travels at 60 mph for 2.5 hours, then at 45 mph for 1.5 hours, what is the total distance?",
]


def run_benchmark(model, tokenizer, gov, device, max_tokens, prompts):
    """Run prompts and report."""
    for i, prompt_text in enumerate(prompts):
        prompt = f"{prompt_text}\n\n"
        print(f"\n{'─'*60}")
        print(f"  PROMPT {i+1}: {prompt_text[:60]}...")
        print(f"{'─'*60}")

        gov.reset()
        t0 = time.time()
        response = generate(model, tokenizer, prompt, device, max_tokens)
        elapsed = time.time() - t0

        words = len(response.split())
        print(f"  [{words} words, {elapsed:.1f}s]")
        print(f"  {response[:400]}{'...' if len(response) > 400 else ''}")

        gov.print_report()


def run_chat(model, tokenizer, gov, device, max_tokens):
    """Interactive chat with adaptive governor."""
    print("\n  Adaptive Governor Chat")
    print("  Type 'quit' to exit, 'report' for governor stats, 'reset' to reset governor\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user.lower() == "quit":
            break
        if user.lower() == "report":
            gov.print_report()
            continue
        if user.lower() == "reset":
            gov.reset()
            print("  Governor reset.\n")
            continue
        if not user:
            continue

        prompt = f"{user}\n\n"
        t0 = time.time()
        response = generate(model, tokenizer, prompt, device, max_tokens)
        elapsed = time.time() - t0
        words = len(response.split())

        print(f"\nAssistant [{words}w, {elapsed:.1f}s]: {response}\n")

        r = gov.report()
        if r["triggers_fired"] > 0:
            print(f"  ! {r['triggers_fired']} triggers fired")
        active = sum(1 for s in r["final_scales"].values() if abs(s - 1.0) > 0.01)
        print(f"  Governor: {r['windows_processed']} windows, {active} layers adjusted\n")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Governor Demo")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mode", default="compass", choices=["compass", "signal", "chat"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--window", type=int, default=32, help="Signal window size")
    parser.add_argument("--lr", type=float, default=0.1, help="Optimization learning rate")
    parser.add_argument("--profile", default=None,
                        help="Path to target profile JSON (for compass mode)")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print("=" * 60)
    print("  ADAPTIVE GOVERNOR")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Mode:       {args.mode}")
    print(f"  Window:     {args.window} tokens")
    print(f"  LR:         {args.lr}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 60)

    # Load model
    print("\n  Loading model...")
    model = load_model(args.model, args.device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Configure
    cfg = AdaptiveConfig(
        window_size=args.window,
        learning_rate=args.lr,
    )

    # Create governor
    if args.mode == "signal":
        gov = AdaptiveGovernor.signal_only(model, tokenizer, config=cfg)
        print("  Mode: signal-only (no target profile)")
    else:
        profile_path = args.profile or "profiles/qwen35_9b_reasoning_dilution_survival_0.8.json"
        if os.path.exists(profile_path):
            gov = AdaptiveGovernor.from_profile(model, tokenizer, profile_path, config=cfg)
            print(f"  Mode: compass -> {profile_path}")
        else:
            print(f"  ! Profile not found: {profile_path}, falling back to signal-only")
            gov = AdaptiveGovernor.signal_only(model, tokenizer, config=cfg)

    if args.mode == "chat":
        run_chat(model, tokenizer, gov, args.device, args.max_tokens)
    else:
        run_benchmark(model, tokenizer, gov, args.device, args.max_tokens, PROMPTS)

    gov.detach()
    print("\n  Done!")


if __name__ == "__main__":
    main()
