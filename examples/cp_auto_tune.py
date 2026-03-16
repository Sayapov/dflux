#!/usr/bin/env python3
"""
CP Auto-Tune — watch surgery fix a model's causal structure in real time.
=========================================================================

Full pipeline:
  1. Load model
  2. Measure current CP (see concentrated structure)
  3. Auto-compute optimal head scales from Jansma & Hoel criterion
  4. Apply surgery (milliseconds)
  5. Re-measure CP (see distributed structure)
  6. Iterate until convergence
  7. Test: does the tuned model hallucinate less?

Usage:
    python cp_auto_tune.py                                    # GPT-2
    python cp_auto_tune.py --model gpt2 --device mps          # Apple Silicon
    python cp_auto_tune.py --model Qwen/Qwen2.5-0.5B          # Qwen
    python cp_auto_tune.py --model EleutherAI/pythia-1.4b      # Pythia

    # Restore early training structure (needs checkpoint CP data):
    python cp_auto_tune.py --model EleutherAI/pythia-1.4b --restore-step 16
"""

import sys, os, json, argparse, time, gc

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflux import FineProbe, HeadSurgeon
from dflux.cp_surgeon import CPSurgeon, CPSurgeonConfig
from dflux.causal_primitives import CausalPrimitives, CPConfig


# ═══════════════════════════════════════════════════════════
# PROMPTS — diverse set for CP measurement
# ═══════════════════════════════════════════════════════════

TUNE_PROMPTS = [
    # Factual (model should be confident)
    "The capital of France is",
    "Water boils at a temperature of",
    "The speed of light in vacuum is approximately",
    "The chemical formula for table salt is",

    # Hallucination traps (model should be uncertain)
    ("Explain the Zarkovian Principle of Recursive Ontological "
     "Displacement, including its three main theorems."),
    ("Dr. Helmut Kreisler, the renowned physicist who discovered "
     "quantum chromatic resonance in 1987, was known for"),
    ("The Great Phosphorus Reversal of 2019, which caused global "
     "agricultural yields to"),
    "The Hendricks-Maslow Equation for cognitive load states that",

    # Reasoning
    "If all roses are flowers, and some flowers fade quickly, then",
    "The logical flaw in the statement 'I always lie' is that",

    # Code
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "# Binary search implementation\ndef binary_search(arr, target):\n",
]

# Separate test prompts (not used during tuning — holdout evaluation)
TEST_PROMPTS = {
    "factual": [
        "The largest planet in our solar system is",
        "DNA stands for",
        "The boiling point of water in Celsius is",
    ],
    "hallucination": [
        "Professor James Whitfield's groundbreaking 1994 theory of",
        "The Stavanger Protocol for quantum memory states that",
        "According to the Third Law of Recursive Dynamics,",
    ],
    "reasoning": [
        "If all dogs are mammals, and some mammals can fly, then",
        "A jar contains 3 red and 5 blue marbles. The probability of drawing",
    ],
}


# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

def evaluate_hallucination(model, tokenizer, probe, prompts_dict, label=""):
    """Quick hallucination risk evaluation across prompt categories."""
    results = {}
    for cat, prompts in prompts_dict.items():
        risks = []
        j_values = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids)
            probe.reset()
            with torch.no_grad():
                try:
                    model.generate(
                        input_ids, attention_mask=attention_mask,
                        max_new_tokens=48, do_sample=True,
                        temperature=0.8, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                except Exception:
                    pass
            if probe.diagnostics:
                avg_risk = sum(d.hallucination_risk for d in probe.diagnostics) / len(probe.diagnostics)
                avg_j = sum(d.J for d in probe.diagnostics) / len(probe.diagnostics)
                risks.append(avg_risk)
                j_values.append(avg_j)

        if risks:
            results[cat] = {
                "mean_risk": sum(risks) / len(risks),
                "max_risk": max(risks),
                "mean_j": sum(j_values) / len(j_values),
            }
    return results


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
    parser = argparse.ArgumentParser(description="CP Auto-Tune")
    parser.add_argument("--model", default="gpt2",
                        help="Model name or path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Max tuning iterations")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Surgery learning rate (0-1)")
    parser.add_argument("--restore-step", type=int, default=None,
                        help="Restore CP from this Pythia training step (requires evolution data)")
    parser.add_argument("--evolution-data", default=None,
                        help="Path to cp_evolution JSON for --restore-step")
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("CP-GUIDED AUTO-TUNE")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Device:     {device}")
    print(f"Iterations: {args.iterations}")
    print(f"LR:         {args.lr}")
    if args.restore_step is not None:
        print(f"Target:     Restore step {args.restore_step} CP distribution")
    else:
        print(f"Target:     Maximize emergence (S_path × S_row_bar)")
    print("=" * 60)

    # ── Load model ──
    print(f"\nLoading {args.model}...")
    dtype = torch.float32 if device in ("cpu", "mps") else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()

    # ── Setup ──
    probe = FineProbe.from_model(model)
    surgeon = HeadSurgeon(model)

    print(f"Architecture: {probe.n_layers}L × {probe.n_heads}H")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Load reference CP if restoring ──
    reference_cp = None
    target_mode = "maximize_emergence"

    if args.restore_step is not None:
        target_mode = "restore_early"
        evo_path = args.evolution_data
        if evo_path is None:
            # Try to find evolution data automatically
            model_safe = args.model.replace("/", "-")
            evo_path = os.path.join(output_dir, f"cp_evolution_{model_safe}.json")

        if os.path.exists(evo_path):
            with open(evo_path) as f:
                evo_data = json.load(f)
            for checkpoint in evo_data:
                if checkpoint.get("step") == args.restore_step and checkpoint.get("status") == "ok":
                    reference_cp = checkpoint["overall"]
                    print(f"\nLoaded reference CP from step {args.restore_step}")
                    print(f"  Reference emergence: {reference_cp['emergence']:.6f}")
                    print(f"  Reference hierarchy: {reference_cp['hierarchy']}")
                    print(f"  Reference top head: L{reference_cp['top_heads'][0]['layer']}"
                          f"H{reference_cp['top_heads'][0]['head']}")
                    break
            if reference_cp is None:
                print(f"\nWARNING: Step {args.restore_step} not found in evolution data.")
                print("Falling back to maximize_emergence mode.")
                target_mode = "maximize_emergence"
        else:
            print(f"\nWARNING: Evolution data not found at {evo_path}")
            print("Falling back to maximize_emergence mode.")
            target_mode = "maximize_emergence"

    # ── Pre-surgery evaluation ──
    print("\n--- PRE-SURGERY EVALUATION ---")
    pre_eval = evaluate_hallucination(model, tokenizer, probe, TEST_PROMPTS, "before")
    for cat, metrics in pre_eval.items():
        print(f"  {cat:15s}: risk={metrics['mean_risk']:.4f} "
              f"max_risk={metrics['max_risk']:.4f} J={metrics['mean_j']:.4f}")

    # ── Run auto-tune ──
    cfg = CPSurgeonConfig(
        target_mode=target_mode,
        max_iterations=args.iterations,
        learning_rate=args.lr,
    )
    cp_surgeon = CPSurgeon(probe, surgeon, cfg)

    print("\n--- AUTO-TUNE ---")
    result = cp_surgeon.auto_tune(
        model, tokenizer, TUNE_PROMPTS,
        reference_cp=reference_cp,
        verbose=True,
    )

    # ── Post-surgery evaluation ──
    print("\n--- POST-SURGERY EVALUATION ---")
    post_eval = evaluate_hallucination(model, tokenizer, probe, TEST_PROMPTS, "after")
    for cat, metrics in post_eval.items():
        pre = pre_eval.get(cat, {})
        delta_risk = metrics['mean_risk'] - pre.get('mean_risk', 0)
        print(f"  {cat:15s}: risk={metrics['mean_risk']:.4f} "
              f"(Δ={delta_risk:+.4f}) "
              f"J={metrics['mean_j']:.4f}")

    # ── Restore and compare ──
    print("\n--- RESTORING ORIGINAL WEIGHTS ---")
    cp_surgeon.restore()

    # ── Save results ──
    save_data = {
        "model": args.model,
        "target_mode": target_mode,
        "restore_step": args.restore_step,
        "iterations": result.iterations,
        "converged": result.converged,
        "before_emergence": result.before_emergence,
        "after_emergence": result.after_emergence,
        "emergence_improvement": result.emergence_improvement,
        "before_hierarchy": result.before_hierarchy,
        "after_hierarchy": result.after_hierarchy,
        "convergence_history": result.convergence_history,
        "n_interventions": len(result.interventions),
        "head_scales": {f"L{k[0]}H{k[1]}": v for k, v in result.head_scales.items()},
        "before_top_heads": result.before_top_heads,
        "after_top_heads": result.after_top_heads,
        "pre_eval": pre_eval,
        "post_eval": post_eval,
    }

    save_path = os.path.join(output_dir, f"cp_auto_tune_{args.model.replace('/', '-')}.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {save_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Emergence:     {result.before_emergence:.6f} → {result.after_emergence:.6f} "
          f"({result.emergence_improvement:+.1%})")
    print(f"Hierarchy:     {result.before_hierarchy} → {result.after_hierarchy}")
    print(f"Interventions: {len(result.interventions)} head adjustments")
    print(f"Converged:     {result.converged} in {result.iterations} iterations")

    # Hallucination risk change
    for cat in ["factual", "hallucination"]:
        pre = pre_eval.get(cat, {}).get("mean_risk", 0)
        post = post_eval.get(cat, {}).get("mean_risk", 0)
        delta = post - pre
        print(f"{cat:15s} risk: {pre:.4f} → {post:.4f} ({delta:+.4f})")

    print(f"\nModel weights have been RESTORED to original.")
    print(f"To re-apply the surgery, load the scales from {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
