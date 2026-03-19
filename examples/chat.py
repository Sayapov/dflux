"""Interactive chat with a D-Flux governed Qwen3.5-9B model.

Usage:
    python examples/chat.py                                    # base model, no governor
    python examples/chat.py --profile profiles/qwen35_9b_reasoning_dilution_survival_1.0.json
    python examples/chat.py --model Qwen/Qwen3.5-9B           # instruct model
    python examples/chat.py --profile ... --telemetry          # with live telemetry dashboard
"""

import argparse
import json
import gc
import sys

import torch
from transformers import AutoTokenizer, AutoConfig


def _promote_qwen35_config(config):
    """Copy text_config fields to top-level config for Qwen3.5."""
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
    """Load Qwen3.5 model, handling nested keys if needed."""
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


def install_governor(model, profile_path):
    """Install scale hooks from a profile JSON."""
    from dflux.multiscale_telemetry import MultiScaleTelemetry

    with open(profile_path) as f:
        profile = json.load(f)

    scales = {int(k): v for k, v in profile["scales"].items()}
    layers = MultiScaleTelemetry._find_transformer_layers(model)
    device = next(model.parameters()).device
    hooks = []

    for i, layer in enumerate(layers):
        scale_val = scales.get(i, 1.0)
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
        arrow = "↑" if scale_val > 1.0 else "↓"
        print(f"    L{i:2d}: {scale_val:.4f} {arrow}")

    print(f"  Installed {len(hooks)} governor hooks\n")
    return hooks


def print_telemetry_dashboard(telem, tokenizer):
    """Print a compact telemetry dashboard after generation."""
    snap = telem.snapshot()
    if not snap:
        print("  [telemetry] No snapshots captured")
        return

    agg = telem.aggregate()
    n_tokens = agg.get("n_tokens", 0)
    n_layers = agg.get("n_layers", 0)

    print()
    print("=" * 72)
    print(f"  TELEMETRY  |  {n_tokens} tokens  |  {n_layers} layers")
    print("=" * 72)

    # ── Residual norms (mean across tokens) ──
    res_norms = agg.get("residual_norms_mean")
    if res_norms:
        print(f"\n  Residual Norms (mean across {n_tokens} tokens):")
        bar_max = max(res_norms)
        for i, val in enumerate(res_norms):
            bar_len = int(40 * val / bar_max) if bar_max > 0 else 0
            layer_type = ""
            if snap.get("layer_types"):
                lt = snap["layer_types"][i]
                layer_type = " [FA]" if lt == "full_attention" else " [LA]"
            print(f"    L{i:2d}{layer_type:5s} {val:8.1f} {'#' * bar_len}")

    # ── Dilution survival (mean across tokens) ──
    ds_mean = agg.get("dilution_survival_mean")
    if ds_mean:
        print(f"\n  Dilution Survival (mean across {n_tokens} tokens):")
        print(f"  {'Layer':>7s}  {'Survival':>9s}  {'Bar':40s}")
        for i, val in enumerate(ds_mean):
            bar_len = int(40 * abs(val))
            sign = "+" if val >= 0 else "-"
            layer_type = ""
            if snap.get("layer_types"):
                lt = snap["layer_types"][i]
                layer_type = " [FA]" if lt == "full_attention" else " [LA]"
            print(f"    L{i:2d}{layer_type:5s} {sign}{abs(val):.4f}  {'|' * bar_len}")

    # ── Entropy cascade (mean) ──
    ent_mean = agg.get("logit_lens_entropy_mean")
    if ent_mean:
        print(f"\n  Logit Lens Entropy (mean across {n_tokens} tokens):")
        ent_max = max(ent_mean) if ent_mean else 1
        for i, val in enumerate(ent_mean):
            bar_len = int(40 * val / ent_max) if ent_max > 0 else 0
            layer_type = ""
            if snap.get("layer_types"):
                lt = snap["layer_types"][i]
                layer_type = " [FA]" if lt == "full_attention" else " [LA]"
            print(f"    L{i:2d}{layer_type:5s} {val:6.2f} {'#' * bar_len}")

    # ── Attention vs MLP energy ratio ──
    attn_norms = agg.get("attn_norms_mean")
    mlp_norms = agg.get("mlp_norms_mean")
    if attn_norms and mlp_norms:
        print(f"\n  Attn vs MLP Energy (mean across {n_tokens} tokens):")
        print(f"  {'Layer':>7s}  {'Attn':>8s}  {'MLP':>8s}  {'Ratio':>6s}")
        for i in range(len(attn_norms)):
            a, m = attn_norms[i], mlp_norms[i]
            ratio = a / m if m > 0 else float('inf')
            layer_type = ""
            if snap.get("layer_types"):
                lt = snap["layer_types"][i]
                layer_type = " [FA]" if lt == "full_attention" else " [LA]"
            print(f"    L{i:2d}{layer_type:5s} {a:8.1f}  {m:8.1f}  {ratio:6.2f}")

    # ── Last token's top predictions at each layer ──
    top_tokens = snap.get("logit_lens_top_tokens")
    if top_tokens and tokenizer:
        print(f"\n  Logit Lens: Top prediction at each layer (last token):")
        for i, layer_preds in enumerate(top_tokens):
            if layer_preds and len(layer_preds) > 0:
                tok_id, prob = layer_preds[0]
                tok_str = tokenizer.decode([tok_id]).strip()[:15]
                layer_type = ""
                if snap.get("layer_types"):
                    lt = snap["layer_types"][i]
                    layer_type = " [FA]" if lt == "full_attention" else " [LA]"
                bar_len = int(30 * prob)
                print(f"    L{i:2d}{layer_type:5s} {prob:5.1%} {tok_str:>15s}  {'#' * bar_len}")

    print()
    print("=" * 72)
    print()


def chat_loop(model, tokenizer, device, max_new_tokens=512, telem=None):
    """Simple interactive chat loop."""
    print("=" * 60)
    print("  D-Flux Chat  |  type 'quit' to exit")
    if telem:
        print("  Telemetry: ON")
    print("=" * 60)
    print()

    history = []

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

        # Clear telemetry from previous turn
        if telem:
            telem.reset()

        # For base models, just do completion. For instruct, use chat format.
        if tokenizer.chat_template:
            history.append({"role": "user", "content": user_input})
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
        else:
            # Base model — use a simple Q/A format
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

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"\nModel: {response}\n")

        if tokenizer.chat_template:
            history.append({"role": "assistant", "content": response})

        # Print telemetry dashboard after response
        if telem:
            print_telemetry_dashboard(telem, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="Chat with a D-Flux governed model")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base",
                        help="HuggingFace model name")
    parser.add_argument("--profile", default=None,
                        help="Path to governor profile JSON")
    parser.add_argument("--device", default="mps",
                        help="Device (mps, cuda, cpu)")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Model dtype")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens per response")
    parser.add_argument("--telemetry", action="store_true",
                        help="Enable live telemetry dashboard after each response")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"\n  Model:    {args.model}")
    print(f"  Profile:  {args.profile or 'none (baseline)'}")
    print(f"  Device:   {args.device}")
    print(f"  Dtype:    {args.dtype}")
    print(f"  Telemetry: {'ON' if args.telemetry else 'OFF'}\n")

    # Load model
    model = load_model(args.model, args.device, dtype)

    # Install governor if profile provided
    if args.profile:
        print(f"  Applying governor profile: {args.profile}")
        install_governor(model, args.profile)

    # Attach telemetry if requested
    telem = None
    if args.telemetry:
        from dflux.multiscale_telemetry import MultiScaleTelemetry, TelemetryConfig
        cfg = TelemetryConfig(
            logit_lens=True,
            cross_layer=False,       # skip for speed
            mlp_internals=True,
            entropy_cascade=True,
            outlier_detection=False,  # skip for speed
            capture_residuals=False,
            max_snapshots=2048,
        )
        tokenizer_for_telem = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        telem = MultiScaleTelemetry.from_model(model, tokenizer_for_telem, cfg=cfg)
        print("  Telemetry attached!\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Go
    chat_loop(model, tokenizer, args.device, args.max_tokens, telem=telem)


if __name__ == "__main__":
    main()
