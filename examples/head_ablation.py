"""Per-head ablation study for Qwen3.5-9B.

Zeros out one head at a time and runs the same prompt to see what changes.
Shows exactly what each head contributes to the output.

Usage:
    python examples/head_ablation.py --device mps --dtype bfloat16
    python examples/head_ablation.py --prompt "your question here"
    python examples/head_ablation.py --heads 3:11,7:5,7:4,31:0  # specific heads only
"""

import argparse
import json
import gc
import os

import torch
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


# ── Ablation hook system ──────────────────────────────────────────

QWEN35_FA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


class HeadAblator:
    """Zeros out a single head in a single layer during forward pass."""

    def __init__(self, model, num_heads):
        self.model = model
        self.num_heads = num_heads
        self.patched_layers = {}  # layer_idx -> (attn_module, original_forward)
        self.active_ablation = None  # (layer_idx, head_idx) or None

        from dflux.multiscale_telemetry import MultiScaleTelemetry
        self.layers = MultiScaleTelemetry._find_transformer_layers(model)

        # Patch all FA layers once
        for layer_idx in QWEN35_FA_LAYERS:
            if layer_idx >= len(self.layers):
                continue
            layer = self.layers[layer_idx]
            attn = MultiScaleTelemetry._find_attn_module(layer)
            if attn is None or not hasattr(attn, 'o_proj'):
                continue

            self._patch_layer(layer_idx, attn)

    def _patch_layer(self, layer_idx, attn):
        original_forward = attn.forward
        ablator = self

        def make_hooked_forward(attn_mod, li):
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

                # ══ ABLATION: zero out the target head ══
                if ablator.active_ablation is not None:
                    abl_layer, abl_head = ablator.active_ablation
                    if abl_layer == li:
                        attn_output[:, :, abl_head, :] = 0.0

                # Normal forward continues
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate_raw)
                attn_output = attn_mod.o_proj(attn_output)

                return attn_output, attn_weights

            return hooked_forward

        attn.forward = make_hooked_forward(attn, layer_idx)
        self.patched_layers[layer_idx] = (attn, original_forward)

    def set_ablation(self, layer_idx, head_idx):
        """Set which head to ablate. None to disable."""
        self.active_ablation = (layer_idx, head_idx) if layer_idx is not None else None

    def clear_ablation(self):
        self.active_ablation = None


def generate_response(model, tokenizer, prompt_text, device, max_tokens=256):
    """Generate a response and return it."""
    prompt = f"Question: {prompt_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy for reproducibility
            temperature=1.0,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


# ── Main ──────────────────────────────────────────────────────────

# Heads to ablate — the interesting ones from our analysis
DEFAULT_HEADS = [
    (3, 0, "task-sensitive"),
    (3, 1, "task-sensitive"),
    (3, 2, "task-sensitive"),
    (3, 3, "task-sensitive"),
    (3, 11, "topic tracker, highest CV in L3"),
    (7, 4, "identity/self-reinforcement head"),
    (7, 5, "#1 task-sensitive, BOS-anchor"),
    (7, 7, "moderately sensitive"),
    (7, 13, "high norm"),
    (11, 14, "task-sensitive, BOS-anchor"),
    (15, 9, "task-sensitive, diffuse"),
    (19, 3, "infrastructure (stable)"),
    (23, 3, "infrastructure (stable)"),
    (27, 3, "infrastructure (stable)"),
    (31, 0, "convergence/entity head"),
    (31, 3, "infrastructure, high gate at output"),
]


def main():
    parser = argparse.ArgumentParser(description="Per-head ablation study")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per generation (keep low for speed)")
    parser.add_argument("--heads", default=None,
                        help="Specific heads to ablate, e.g. '3:11,7:5,31:0'")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    prompt_text = args.prompt or (
        "What is the temperature on the far side of the moon, "
        "and what would happen to a cucumber if it suddenly appeared there."
    )

    # Parse heads
    if args.heads:
        heads_to_test = []
        for pair in args.heads.split(","):
            l, h = pair.strip().split(":")
            heads_to_test.append((int(l), int(h), "user-selected"))
    else:
        heads_to_test = DEFAULT_HEADS

    print(f"\n  Per-Head Ablation Study")
    print(f"  Model:      {args.model}")
    print(f"  Heads:      {len(heads_to_test)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Prompt:     {prompt_text[:60]}...\n")

    # Load model
    model = load_model(args.model, args.device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    text_cfg = getattr(model.config, "text_config", model.config)
    num_heads = text_cfg.num_attention_heads

    # Install ablation hooks
    ablator = HeadAblator(model, num_heads)

    # ── Baseline (no ablation) ──
    print("=" * 72)
    print("  BASELINE (no ablation)")
    print("=" * 72)
    ablator.clear_ablation()
    baseline = generate_response(model, tokenizer, prompt_text, args.device, args.max_tokens)
    baseline_len = len(baseline.split())
    print(f"\n  [{baseline_len} words]\n")
    # Show first 200 chars
    print(f"  {baseline[:300]}{'...' if len(baseline) > 300 else ''}\n")

    # ── Ablate each head ──
    results = [{"head": "baseline", "response": baseline, "words": baseline_len}]

    for layer_idx, head_idx, label in heads_to_test:
        print("─" * 72)
        print(f"  ABLATING L{layer_idx}:H{head_idx:02d} — {label}")
        print("─" * 72)

        ablator.set_ablation(layer_idx, head_idx)
        response = generate_response(model, tokenizer, prompt_text, args.device, args.max_tokens)
        resp_len = len(response.split())

        # Compare to baseline
        len_delta = resp_len - baseline_len
        len_pct = (len_delta / baseline_len * 100) if baseline_len > 0 else 0

        # Simple similarity: check if first 100 chars match
        first_match = baseline[:100] == response[:100]
        status = "SAME START" if first_match else "DIFFERENT"

        print(f"  [{resp_len} words, {len_delta:+d} ({len_pct:+.0f}%)] {status}")
        print(f"  {response[:300]}{'...' if len(response) > 300 else ''}\n")

        results.append({
            "head": f"L{layer_idx}:H{head_idx:02d}",
            "label": label,
            "response": response,
            "words": resp_len,
            "words_delta": len_delta,
            "same_start": first_match,
        })

    # ── Summary ──
    print("\n" + "=" * 72)
    print("  ABLATION SUMMARY")
    print("=" * 72)
    print(f"  {'Head':>12s}  {'Words':>6s}  {'Delta':>7s}  {'Start':>6s}  Label")
    print(f"  {'─' * 65}")
    print(f"  {'baseline':>12s}  {baseline_len:>6d}  {'':>7s}  {'':>6s}")

    for r in results[1:]:
        delta_str = f"{r['words_delta']:+d}"
        start_str = "SAME" if r["same_start"] else "DIFF"
        print(f"  {r['head']:>12s}  {r['words']:>6d}  {delta_str:>7s}  {start_str:>6s}  {r.get('label', '')}")

    # ── Save ──
    os.makedirs("data/per_head", exist_ok=True)
    save_path = "data/per_head/ablation_results.json"

    save_data = {
        "model": args.model,
        "prompt": prompt_text,
        "max_tokens": args.max_tokens,
        "results": results,
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Saved to {save_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
