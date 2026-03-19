"""Per-head governor for Qwen3.5-9B.

Applies per-head scale factors to individual attention heads before o_proj.
Can run benchmarks or interactive chat with per-head profiles.

Usage:
    # Generate profile from explorer data
    python examples/per_head_governor.py --generate-profile

    # Run benchmark with per-head profile
    python examples/per_head_governor.py --benchmark --tasks arc_challenge

    # Interactive chat
    python examples/per_head_governor.py --chat --max-tokens 2048

    # Compare per-head vs per-layer profiles
    python examples/per_head_governor.py --benchmark --tasks arc_challenge --compare
"""

import argparse
import json
import gc
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

# ── Reuse model loading ──────────────────────────────────────────

def _promote_qwen35_config(config):
    _SKIP = {"use_return_dict", "output_hidden_states", "output_attentions",
             "torchscript", "pruned_heads", "is_encoder_decoder"}
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        return
    for attr in dir(text_cfg):
        if attr.startswith("_") or attr in _SKIP:
            continue
        try:
            object.__getattribute__(config, attr)
        except AttributeError:
            try:
                val = getattr(text_cfg, attr)
                if not callable(val):
                    setattr(config, attr, val)
            except Exception:
                pass


def _detect_nested_keys(model_name: str) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        idx_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(idx_path) as f:
            keys = json.load(f)["weight_map"].keys()
        return any("language_model." in k for k in keys)
    except Exception:
        pass
    try:
        from safetensors import safe_open
        path = hf_hub_download(model_name, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            keys = f.keys()
        return any("language_model." in k for k in keys)
    except Exception:
        return False


def load_model(model_name, device, dtype):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_qwen35 = getattr(config, "model_type", "") == "qwen3_5"

    if is_qwen35 and _detect_nested_keys(model_name):
        from transformers import Qwen3_5ForCausalLM
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as safe_load

        print(f"  Loading {model_name} with key remapping...")
        _promote_qwen35_config(config)
        model = Qwen3_5ForCausalLM(config).to(dtype=dtype)

        try:
            idx = hf_hub_download(model_name, "model.safetensors.index.json")
            with open(idx) as f:
                weight_map = json.load(f)["weight_map"]
            full_sd = {}
            for shard in set(weight_map.values()):
                path = hf_hub_download(model_name, shard)
                for k, v in safe_load(path, device="cpu").items():
                    full_sd[k.replace("model.language_model.", "model.")] = v
        except Exception:
            path = hf_hub_download(model_name, "model.safetensors")
            full_sd = {k.replace("model.language_model.", "model."): v
                       for k, v in safe_load(path, device="cpu").items()}

        model.load_state_dict(full_sd, strict=False)
        del full_sd
        gc.collect()
        model = model.to(device)
    else:
        from transformers import AutoModelForCausalLM
        if is_qwen35:
            _promote_qwen35_config(config)
        print(f"  Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=dtype,
            trust_remote_code=True, device_map=device
        )

    model.eval()
    return model


# ── Per-head profile generation ───────────────────────────────────

QWEN35_FA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


def generate_profile_from_exploration(explorer_json_path: str, blend: float = 0.8) -> dict:
    """Generate a per-head profile from explorer results.

    Strategy: Use coefficient of variation (CV) as the signal.
    Heads with high CV are task-sensitive — they adapt to content.
    Boost those heads, leave infrastructure heads alone.

    Also factor in: gate values (model's own opinion of head importance),
    and mean norms (absolute contribution).
    """
    with open(explorer_json_path) as f:
        data = json.load(f)

    analysis = data["analysis"]
    num_heads = data["num_heads"]
    fa_layers = data["fa_layers"]

    # Build per-head scale profile
    scales = {}

    for layer_idx in fa_layers:
        layer_key = str(layer_idx)
        cv_vals = analysis["cv"].get(layer_key, [1.0] * num_heads)
        mean_norms = analysis["mean_norms"].get(layer_key, [1.0] * num_heads)

        # Normalize CV within this layer to get relative sensitivity
        cv_tensor = torch.tensor(cv_vals)
        mn_tensor = torch.tensor(mean_norms)

        # CV-based scale: higher CV = more task-sensitive = boost more
        # Normalize: mean CV maps to 1.0, higher CV maps above 1.0
        cv_mean = cv_tensor.mean()
        if cv_mean > 0:
            raw_scales = cv_tensor / cv_mean  # centered around 1.0
        else:
            raw_scales = torch.ones_like(cv_tensor)

        # Apply blend: interpolate between 1.0 and target
        blended = 1.0 + blend * (raw_scales - 1.0)

        # Clamp
        blended = blended.clamp(min=0.5, max=2.0)

        head_scales = {}
        for h in range(num_heads):
            s = round(blended[h].item(), 4)
            head_scales[str(h)] = s

        scales[str(layer_idx)] = head_scales

    profile = {
        "type": "per_head",
        "signal": "cv_sensitivity",
        "blend": blend,
        "num_heads": num_heads,
        "fa_layers": fa_layers,
        "scales": scales,
    }

    return profile


def generate_profile_manual() -> dict:
    """Hand-crafted profile based on attention pattern analysis.

    Informed by the head_attention_viewer results:
    - L3:H11 — topic tracker, diffuse gatherer → boost
    - L7:H05 — BOS-anchor/task head, most task-sensitive → boost
    - L7:H04 — identity/reinforcement head → slight boost
    - L11:H14 — BOS-anchor, task-sensitive → boost
    - L15:H09 — task-sensitive, diffuse → boost
    - L31:H00 — convergence/entity head → boost
    - Infrastructure H03 chain → leave at 1.0
    - Low-gate heads in L3 → leave alone (model suppresses them for a reason)
    """
    num_heads = 16
    fa_layers = QWEN35_FA_LAYERS

    # Start everything at 1.0
    scales = {}
    for layer_idx in fa_layers:
        scales[str(layer_idx)] = {str(h): 1.0 for h in range(num_heads)}

    # ── Targeted adjustments ──

    # L3: Topic tracking heads — gentle boost (gates are nearly closed,
    # so we need to push through the gate suppression)
    scales["3"]["11"] = 1.4   # topic tracker — highest CV in L3
    scales["3"]["3"] = 1.2    # task-sensitive
    scales["3"]["1"] = 1.15   # task-sensitive
    scales["3"]["2"] = 1.15   # task-sensitive
    scales["3"]["0"] = 1.1    # task-sensitive

    # L7: Task identification heads — strongest signal
    scales["7"]["5"] = 1.5    # #1 most task-sensitive, BOS-anchor/task reader
    scales["7"]["4"] = 1.3    # identity/reinforcement head
    scales["7"]["7"] = 1.15   # moderately sensitive

    # L11: Mid-network boosters
    scales["11"]["14"] = 1.3  # task-sensitive, BOS-anchor
    scales["11"]["1"] = 1.15  # moderate sensitivity
    scales["11"]["11"] = 1.15 # moderate sensitivity

    # L15: Single targeted boost
    scales["15"]["9"] = 1.25  # task-sensitive, diffuse attention

    # L19-L23: Infrastructure — mostly leave alone
    # These are stable and doing their job

    # L31: Output convergence — careful tuning
    scales["31"]["0"] = 1.3   # convergence/entity gathering head
    scales["31"]["3"] = 1.1   # infrastructure but high gate at output

    profile = {
        "type": "per_head",
        "signal": "manual_attention_analysis",
        "blend": "manual",
        "num_heads": num_heads,
        "fa_layers": fa_layers,
        "scales": scales,
    }

    return profile


# ── Per-head hook installation ────────────────────────────────────

def install_per_head_governor(model, profile: dict):
    """Install per-head scale hooks by monkey-patching attention forward."""
    from dflux.multiscale_telemetry import MultiScaleTelemetry

    layers = MultiScaleTelemetry._find_transformer_layers(model)
    device = next(model.parameters()).device
    num_heads = profile["num_heads"]
    hooks_installed = 0

    for layer_idx_str, head_scales in profile["scales"].items():
        layer_idx = int(layer_idx_str)
        if layer_idx >= len(layers):
            continue

        # Check if any head in this layer needs scaling
        scale_vals = [head_scales.get(str(h), 1.0) for h in range(num_heads)]
        if all(abs(s - 1.0) < 1e-6 for s in scale_vals):
            continue

        layer = layers[layer_idx]
        attn = MultiScaleTelemetry._find_attn_module(layer)
        if attn is None or not hasattr(attn, 'o_proj'):
            continue

        # Build scale tensor: [1, 1, num_heads, 1] for broadcasting
        scale_tensor = torch.tensor(scale_vals, device=device, dtype=torch.float32)
        scale_tensor = scale_tensor.view(1, 1, num_heads, 1)  # [1, 1, H, 1]

        def make_hooked_forward(attn_mod, head_scale_t):
            original_forward = attn_mod.forward

            def hooked_forward(hidden_states, position_embeddings,
                               attention_mask=None, past_key_values=None,
                               **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_mod.head_dim)

                query_states, gate = torch.chunk(
                    attn_mod.q_proj(hidden_states).view(*input_shape, -1, attn_mod.head_dim * 2),
                    2, dim=-1
                )
                gate_raw = gate.reshape(*input_shape, -1)

                query_states = attn_mod.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                from transformers.models.qwen3_5.modeling_qwen3_5 import (
                    apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS,
                    eager_attention_forward
                )
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, attn_mod.layer_idx
                    )

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    attn_mod.config._attn_implementation, eager_attention_forward
                )

                attn_output, attn_weights = attention_interface(
                    attn_mod,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not attn_mod.training else attn_mod.attention_dropout,
                    scaling=attn_mod.scaling,
                    **kwargs,
                )

                # ══ PER-HEAD SCALING ══
                # attn_output shape: [batch, seq, num_heads, head_dim]
                attn_output = attn_output * head_scale_t.to(attn_output.dtype)

                # Continue with normal forward
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate_raw)
                attn_output = attn_mod.o_proj(attn_output)

                return attn_output, attn_weights

            return hooked_forward

        attn.forward = make_hooked_forward(attn, scale_tensor)
        hooks_installed += 1

        # Print active scales for this layer
        active = [(h, s) for h, s in enumerate(scale_vals) if abs(s - 1.0) > 1e-6]
        active_str = " ".join(
            f"H{h}:{'↑' if s > 1 else '↓'}{s:.2f}" for h, s in active
        )
        print(f"    L{layer_idx:2d}: {active_str}")

    print(f"  Installed per-head governor on {hooks_installed} layers\n")


# ── Benchmark runner ──────────────────────────────────────────────

def run_benchmark(model, tokenizer, tasks, device, dtype):
    """Run lm-eval benchmark using programmatic API."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    # Wrap model in HFLM-compatible interface
    lm = HFLM(pretrained=model, tokenizer=tokenizer, dtype=str(dtype).replace("torch.", ""))

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks.split(","),
        batch_size=4,
        device=str(device),
    )

    return results


# ── Chat loop ─────────────────────────────────────────────────────

def chat_loop(model, tokenizer, device, max_new_tokens=512):
    print("=" * 60)
    print("  D-Flux Per-Head Governor Chat  |  type 'quit' to exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        prompt = f"Question: {user_input}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"\nModel: {response}\n")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-head governor")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--profile", default=None,
                        help="Path to per-head profile JSON (default: auto-generate manual)")
    parser.add_argument("--generate-profile", action="store_true",
                        help="Generate and save per-head profiles, then exit")
    parser.add_argument("--blend", type=float, default=0.8,
                        help="Blend for CV-based profile generation")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run lm-eval benchmark")
    parser.add_argument("--tasks", default="arc_challenge",
                        help="Comma-separated benchmark tasks")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    # ── Profile generation mode ──
    if args.generate_profile:
        print("\n  Generating per-head profiles...\n")

        # Manual profile from attention analysis
        manual = generate_profile_manual()
        manual_path = "profiles/qwen35_9b_per_head_manual.json"
        os.makedirs("profiles", exist_ok=True)
        with open(manual_path, "w") as f:
            json.dump(manual, f, indent=2)
        print(f"  Saved manual profile to {manual_path}")

        # Print what it does
        for layer_key, heads in manual["scales"].items():
            active = {h: s for h, s in heads.items() if abs(float(s) - 1.0) > 1e-6}
            if active:
                parts = " ".join(f"H{h}:{s}" for h, s in active.items())
                print(f"    L{layer_key}: {parts}")

        # CV-based profile from explorer data
        explorer_path = "data/per_head/per_head_exploration.json"
        if os.path.exists(explorer_path):
            cv_profile = generate_profile_from_exploration(explorer_path, args.blend)
            cv_path = f"profiles/qwen35_9b_per_head_cv_{args.blend}.json"
            with open(cv_path, "w") as f:
                json.dump(cv_profile, f, indent=2)
            print(f"\n  Saved CV-based profile to {cv_path}")
        else:
            print(f"\n  [SKIP] No explorer data at {explorer_path}")

        print("\n  Done!")
        return

    # ── Load model ──
    print(f"\n  Per-Head Governor")
    print(f"  Model:   {args.model}")
    print(f"  Device:  {args.device}")
    print(f"  Dtype:   {args.dtype}\n")

    model = load_model(args.model, args.device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ── Load or generate profile ──
    if args.profile:
        with open(args.profile) as f:
            profile = json.load(f)
        print(f"  Loaded profile: {args.profile}")
    else:
        print("  Using manual profile from attention analysis")
        profile = generate_profile_manual()

    # ── Install per-head governor ──
    print(f"\n  Installing per-head scales:")
    install_per_head_governor(model, profile)

    # ── Run mode ──
    if args.benchmark:
        print(f"  Running benchmark: {args.tasks}")
        results = run_benchmark(model, tokenizer, args.tasks, args.device, dtype)

        # Extract and print results
        task_results = {}
        for task_name, task_data in results["results"].items():
            task_results[task_name] = {}
            for metric, value in task_data.items():
                if "stderr" not in metric and metric != "alias":
                    task_results[task_name][metric] = value
                    print(f"  {task_name} / {metric}: {value:.4f}")

        # Save
        os.makedirs("data/benchmarks", exist_ok=True)
        save_name = f"per_head_governor_{args.tasks.replace(',', '_')}.json"
        with open(f"data/benchmarks/{save_name}", "w") as f:
            json.dump(task_results, f, indent=2)
        print(f"\n  Saved to data/benchmarks/{save_name}")

    elif args.chat:
        chat_loop(model, tokenizer, args.device, args.max_tokens)

    else:
        print("  Specify --benchmark or --chat")


if __name__ == "__main__":
    main()
