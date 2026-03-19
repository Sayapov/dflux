"""Random profile chaos experiment.

Generates a bunch of wild scale profiles — random, inverted curves, spikes,
sine waves, whatever — and runs each through the same prompt to see what happens.

Usage:
    python examples/random_profiles.py --device mps --dtype bfloat16
    python examples/random_profiles.py --max-tokens 128  # faster
"""

import argparse
import json
import gc
import os
import random
import math

import torch
from transformers import AutoTokenizer, AutoConfig

# ── Model loading (reused) ────────────────────────────────────────

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


# ── Hook system (simple o_proj scaling, same as chat.py) ──────────

def install_scales(model, scales_dict):
    """Install o_proj scale hooks. Returns list of hook handles."""
    from dflux.multiscale_telemetry import MultiScaleTelemetry

    layers = MultiScaleTelemetry._find_transformer_layers(model)
    device = next(model.parameters()).device
    hooks = []

    for i, layer in enumerate(layers):
        scale_val = scales_dict.get(i, 1.0)
        if abs(scale_val - 1.0) < 1e-6:
            continue

        attn = MultiScaleTelemetry._find_attn_module(layer)
        if attn is None:
            continue

        o_proj = None
        for name in ("o_proj", "c_proj", "dense", "out_proj"):
            if hasattr(attn, name):
                o_proj = getattr(attn, name)
                break
        if o_proj is None:
            continue

        scale_tensor = torch.tensor([scale_val], device=device, dtype=torch.float32)

        def _make_hook(s):
            def hook(module, input, output):
                return output * s.to(output.dtype)
            return hook

        h = o_proj.register_forward_hook(_make_hook(scale_tensor))
        hooks.append(h)

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ── Profile generators ────────────────────────────────────────────

FA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
ALL_LAYERS = list(range(32))


def make_profile(name, description, scales_dict):
    return {"name": name, "description": description, "scales": scales_dict}


def generate_profiles():
    """Generate a wild collection of scale profiles."""
    profiles = []

    # ── 0. Baseline (no scaling) ──
    profiles.append(make_profile(
        "baseline", "No scaling — reference",
        {}
    ))

    # ── 1. Our best static profile (0.8 blend) ──
    profiles.append(make_profile(
        "dilution_0.8", "Best known static profile",
        {3: 1.1002, 7: 1.0462, 11: 1.0053, 15: 0.9485, 19: 0.9807, 23: 0.9978, 27: 1.0162, 31: 0.9982}
    ))

    # ── 2. Pure random (FA layers only, 0.5-2.0) ──
    random.seed(42)
    for trial in range(3):
        scales = {l: round(random.uniform(0.5, 2.0), 3) for l in FA_LAYERS}
        profiles.append(make_profile(
            f"random_fa_{trial}", f"Random scales on FA layers (seed {42+trial})",
            scales
        ))

    # ── 3. Pure random (ALL 32 layers, 0.5-2.0) ──
    random.seed(100)
    for trial in range(3):
        scales = {l: round(random.uniform(0.5, 2.0), 3) for l in ALL_LAYERS}
        profiles.append(make_profile(
            f"random_all_{trial}", f"Random scales on ALL layers (seed {100+trial})",
            scales
        ))

    # ── 4. Sine wave (smooth oscillation across layers) ──
    for freq in [0.5, 1.0, 2.0]:
        scales = {}
        for l in ALL_LAYERS:
            scales[l] = round(1.0 + 0.5 * math.sin(2 * math.pi * freq * l / 32), 3)
        profiles.append(make_profile(
            f"sine_{freq}hz", f"Sine wave, {freq} cycles across 32 layers",
            scales
        ))

    # ── 5. Ramp up (early layers low, late layers high) ──
    scales = {l: round(0.5 + 1.5 * (l / 31), 3) for l in ALL_LAYERS}
    profiles.append(make_profile("ramp_up", "Linear ramp: 0.5 at L0 → 2.0 at L31", scales))

    # ── 6. Ramp down (early layers high, late layers low) ──
    scales = {l: round(2.0 - 1.5 * (l / 31), 3) for l in ALL_LAYERS}
    profiles.append(make_profile("ramp_down", "Linear ramp: 2.0 at L0 → 0.5 at L31", scales))

    # ── 7. V-shape (high at edges, low in middle) ──
    scales = {}
    for l in ALL_LAYERS:
        dist = abs(l - 15.5) / 15.5
        scales[l] = round(0.6 + 1.4 * dist, 3)
    profiles.append(make_profile("v_shape", "V-shape: high at L0/L31, low at L15-16", scales))

    # ── 8. Inverted V (low at edges, high in middle) ──
    scales = {}
    for l in ALL_LAYERS:
        dist = 1.0 - abs(l - 15.5) / 15.5
        scales[l] = round(0.6 + 1.4 * dist, 3)
    profiles.append(make_profile("inverted_v", "Inverted V: high at L15-16, low at edges", scales))

    # ── 9. Spike at single layer ──
    for spike_layer in [3, 15, 27]:
        scales = {spike_layer: 3.0}
        profiles.append(make_profile(
            f"spike_L{spike_layer}", f"3x spike at L{spike_layer} only",
            scales
        ))

    # ── 10. Kill a single layer ──
    for kill_layer in [3, 15, 31]:
        scales = {kill_layer: 0.1}
        profiles.append(make_profile(
            f"kill_L{kill_layer}", f"Nearly zero L{kill_layer}",
            scales
        ))

    # ── 11. Only early layers ──
    scales = {l: 1.5 for l in range(8)}
    profiles.append(make_profile("boost_early", "1.5x on layers 0-7 only", scales))

    # ── 12. Only late layers ──
    scales = {l: 1.5 for l in range(24, 32)}
    profiles.append(make_profile("boost_late", "1.5x on layers 24-31 only", scales))

    # ── 13. Only middle layers ──
    scales = {l: 1.5 for l in range(12, 20)}
    profiles.append(make_profile("boost_middle", "1.5x on layers 12-19 only", scales))

    # ── 14. Alternating high/low ──
    scales = {l: 1.5 if l % 2 == 0 else 0.7 for l in ALL_LAYERS}
    profiles.append(make_profile("alternating", "1.5x even layers, 0.7x odd layers", scales))

    # ── 15. Exponential growth ──
    scales = {l: round(0.5 * math.exp(0.045 * l), 3) for l in ALL_LAYERS}
    profiles.append(make_profile("exponential", "Exponential growth 0.5 → ~2.0", scales))

    # ── 16. Only DeltaNet (linear attention) layers ──
    la_layers = [l for l in ALL_LAYERS if l not in FA_LAYERS]
    scales = {l: 1.5 for l in la_layers}
    profiles.append(make_profile("boost_deltanet", "1.5x on all 24 DeltaNet layers, FA at 1.0", scales))

    # ── 17. Suppress DeltaNet, boost FA ──
    scales = {}
    for l in ALL_LAYERS:
        if l in FA_LAYERS:
            scales[l] = 1.5
        else:
            scales[l] = 0.7
    profiles.append(make_profile("fa_up_la_down", "FA layers 1.5x, DeltaNet 0.7x", scales))

    # ── 18. All 2x ──
    scales = {l: 2.0 for l in ALL_LAYERS}
    profiles.append(make_profile("all_2x", "Everything doubled", scales))

    # ── 19. All 0.5x ──
    scales = {l: 0.5 for l in ALL_LAYERS}
    profiles.append(make_profile("all_half", "Everything halved", scales))

    # ── 20. Chaos: random per-head-ish via extreme layer variation ──
    random.seed(999)
    scales = {l: round(random.choice([0.3, 0.5, 1.0, 1.5, 2.0, 3.0]), 3) for l in ALL_LAYERS}
    profiles.append(make_profile("chaos", "Wild random from {0.3, 0.5, 1.0, 1.5, 2.0, 3.0}", scales))

    return profiles


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Random profile chaos experiment")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    prompt_text = args.prompt or (
        "What is the temperature on the far side of the moon, "
        "and what would happen to a cucumber if it suddenly appeared there."
    )

    profiles = generate_profiles()

    print(f"\n  Random Profile Chaos Experiment")
    print(f"  Model:      {args.model}")
    print(f"  Profiles:   {len(profiles)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Prompt:     {prompt_text[:60]}...\n")

    # Load model once
    model = load_model(args.model, args.device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results = []

    for i, profile in enumerate(profiles):
        name = profile["name"]
        desc = profile["description"]
        scales = profile["scales"]

        print(f"\n{'=' * 72}")
        print(f"  [{i+1}/{len(profiles)}] {name}")
        print(f"  {desc}")

        # Show active scales
        active = {k: v for k, v in scales.items() if abs(v - 1.0) > 1e-6}
        if active:
            n_active = len(active)
            sample = list(active.items())[:6]
            sample_str = " ".join(f"L{k}:{v}" for k, v in sample)
            if n_active > 6:
                sample_str += f" ... ({n_active} total)"
            print(f"  Scales: {sample_str}")
        else:
            print(f"  Scales: none (baseline)")

        print(f"{'=' * 72}")

        # Install hooks
        hooks = install_scales(model, scales)

        # Generate
        prompt = f"Question: {prompt_text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=1.0,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        n_words = len(response.split())

        # Remove hooks for next profile
        remove_hooks(hooks)

        # Quick quality heuristics
        has_think = "<think>" in response.lower() or "</think>" in response.lower()
        has_repetition = False
        words = response.split()
        if len(words) > 20:
            # Check for repeated phrases
            for window in range(3, 8):
                for j in range(len(words) - window * 2):
                    chunk1 = " ".join(words[j:j+window])
                    chunk2 = " ".join(words[j+window:j+window*2])
                    if chunk1 == chunk2:
                        has_repetition = True
                        break
                if has_repetition:
                    break

        is_gibberish = n_words < 5 or (n_words > 10 and len(set(words)) < n_words * 0.3)

        quality = "OK"
        if is_gibberish:
            quality = "GIBBERISH"
        elif has_repetition:
            quality = "REPETITIVE"
        elif has_think:
            quality = "THINKS"

        print(f"\n  [{n_words} words] [{quality}]")
        print(f"  {response[:300]}{'...' if len(response) > 300 else ''}")

        results.append({
            "name": name,
            "description": desc,
            "scales": {str(k): v for k, v in scales.items()},
            "response": response,
            "words": n_words,
            "quality": quality,
            "has_think": has_think,
            "has_repetition": has_repetition,
        })

    # ── Summary ──
    print(f"\n\n{'=' * 72}")
    print(f"  CHAOS EXPERIMENT SUMMARY")
    print(f"{'=' * 72}")
    print(f"  {'#':>3s}  {'Name':>20s}  {'Words':>6s}  {'Quality':>10s}  First 80 chars")
    print(f"  {'─' * 70}")

    for i, r in enumerate(results):
        first80 = r["response"][:80].replace("\n", " ")
        print(f"  {i+1:3d}  {r['name']:>20s}  {r['words']:>6d}  {r['quality']:>10s}  {first80}")

    # ── Save ──
    os.makedirs("data/chaos", exist_ok=True)
    save_path = "data/chaos/random_profiles_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to {save_path}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
