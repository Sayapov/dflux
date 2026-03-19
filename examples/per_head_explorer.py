"""Per-head attention explorer for Qwen3.5-9B.

Runs multiple prompts through the model and captures per-head attention
output norms and entropy across all full-attention layers. Visualises
head specialisation patterns.

Usage:
    python examples/per_head_explorer.py
    python examples/per_head_explorer.py --device cuda --dtype float16
    python examples/per_head_explorer.py --max-tokens 64   # faster
"""

import argparse
import json
import gc
import os
import sys
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

# ── Reuse model loading from chat.py ──────────────────────────────

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


# ── Per-head capture hooks ────────────────────────────────────────

# Full-attention layer indices in Qwen3.5 (every 4th, 0-indexed from layer 3)
QWEN35_FA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


class PerHeadCapture:
    """Captures per-head attention output norms and attention weight entropy."""

    def __init__(self):
        self.hooks = []
        self.head_norms: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.head_entropy: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.gate_values: Dict[int, List[torch.Tensor]] = defaultdict(list)

    def install(self, model):
        """Install hooks on full-attention layers by monkey-patching forward."""
        from dflux.multiscale_telemetry import MultiScaleTelemetry
        layers = MultiScaleTelemetry._find_transformer_layers(model)

        for layer_idx in QWEN35_FA_LAYERS:
            if layer_idx >= len(layers):
                continue

            layer = layers[layer_idx]
            attn = MultiScaleTelemetry._find_attn_module(layer)
            if attn is None:
                continue

            # Verify this is a full attention layer with o_proj
            if not hasattr(attn, 'o_proj'):
                continue

            original_forward = attn.forward
            capture = self
            lidx = layer_idx

            def make_hooked_forward(orig_fwd, cap, li):
                def hooked_forward(hidden_states, position_embeddings,
                                   attention_mask=None, past_key_values=None,
                                   **kwargs):
                    # Run up to attention computation manually
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, attn.head_dim)

                    query_states, gate = torch.chunk(
                        attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2),
                        2, dim=-1
                    )
                    gate_raw = gate.reshape(*input_shape, -1)

                    query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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
                            key_states, value_states, attn.layer_idx
                        )

                    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                        attn.config._attn_implementation, eager_attention_forward
                    )

                    # Get attention output and weights
                    attn_output, attn_weights = attention_interface(
                        attn,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        dropout=0.0 if not attn.training else attn.attention_dropout,
                        scaling=attn.scaling,
                        **kwargs,
                    )

                    # ══ CAPTURE per-head norms ══
                    # attn_output shape: [batch, seq, num_heads, head_dim]
                    with torch.no_grad():
                        # Per-head L2 norm: [num_heads]
                        # Mean across batch and sequence
                        per_head_norm = attn_output.float().norm(dim=-1).mean(dim=(0, 1))
                        cap.head_norms[li].append(per_head_norm.cpu())

                        # Per-head attention entropy from weights
                        # attn_weights shape: [batch, num_heads, seq_q, seq_k]
                        if attn_weights is not None:
                            # Shannon entropy of attention distribution per head
                            # Mean across batch and query positions
                            aw = attn_weights.float().clamp(min=1e-10)
                            entropy = -(aw * aw.log()).sum(dim=-1).mean(dim=(0, 2))
                            cap.head_entropy[li].append(entropy.cpu())

                        # Gate values per head
                        gate_sigmoid = torch.sigmoid(gate_raw.float())
                        per_head_gate = gate_sigmoid.mean(dim=(0, 1))
                        cap.gate_values[li].append(per_head_gate.cpu())

                    # Continue with normal forward
                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = attn_output * torch.sigmoid(gate_raw)
                    attn_output = attn.o_proj(attn_output)

                    return attn_output, attn_weights

                return hooked_forward

            attn.forward = make_hooked_forward(original_forward, capture, lidx)
            self.hooks.append((attn, original_forward))
            print(f"    Hooked L{layer_idx} [FA] — {attn.config.num_attention_heads} heads")

        print(f"  Installed {len(self.hooks)} per-head capture hooks\n")

    def reset(self):
        """Clear captured data between prompts."""
        self.head_norms.clear()
        self.head_entropy.clear()
        self.gate_values.clear()

    def aggregate(self) -> Dict[int, dict]:
        """Aggregate captured data into per-layer, per-head means."""
        results = {}
        for layer_idx in sorted(self.head_norms.keys()):
            norms = self.head_norms[layer_idx]
            if not norms:
                continue

            mean_norms = torch.stack(norms).mean(dim=0)

            mean_entropy = None
            if self.head_entropy.get(layer_idx):
                mean_entropy = torch.stack(self.head_entropy[layer_idx]).mean(dim=0)

            mean_gate = None
            if self.gate_values.get(layer_idx):
                mean_gate = torch.stack(self.gate_values[layer_idx]).mean(dim=0)

            results[layer_idx] = {
                "norms": mean_norms,
                "entropy": mean_entropy,
                "gate": mean_gate,
            }

        return results

    def cleanup(self):
        """Restore original forward methods."""
        for attn, orig_fwd in self.hooks:
            attn.forward = orig_fwd
        self.hooks.clear()


# ── Prompts ───────────────────────────────────────────────────────

PROMPTS = {
    "factual": "What is the capital of France and when was it founded?",
    "math": "If a train travels 120km in 2 hours, then 180km in 3 hours, what is its average speed?",
    "creative": "Write a short poem about a robot learning to dance",
    "reasoning": "What is the temperature on the far side of the moon, and what would happen to a cucumber if it suddenly appeared there.",
}


# ── Visualisation ─────────────────────────────────────────────────

def print_heatmap(title: str, data: Dict[int, torch.Tensor], num_heads: int,
                  label: str = "norm"):
    """Print an ASCII heatmap: rows = layers, cols = heads."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")

    # Collect all values for global normalization
    all_vals = torch.cat([v for v in data.values()])
    vmin, vmax = all_vals.min().item(), all_vals.max().item()
    vrange = vmax - vmin if vmax > vmin else 1.0

    # Header
    head_labels = "".join(f"{h:>4d}" for h in range(num_heads))
    print(f"  {'':>6s} {head_labels}")
    print(f"  {'':>6s} {'----' * num_heads}")

    BLOCKS = " ░▒▓█"

    for layer_idx in sorted(data.keys()):
        vals = data[layer_idx]
        row = ""
        for h in range(num_heads):
            v = vals[h].item()
            level = int(4 * (v - vmin) / vrange)
            level = max(0, min(4, level))
            row += f"  {BLOCKS[level]} "
        print(f"  L{layer_idx:2d}:  {row}  [{label}: {vals.min().item():.2f}-{vals.max().item():.2f}]")

    print()


def print_analysis(all_results: Dict[str, Dict[int, dict]], num_heads: int):
    """Print cross-prompt analysis: specialisation and stability."""

    print(f"\n{'=' * 72}")
    print(f"  CROSS-PROMPT ANALYSIS")
    print(f"{'=' * 72}")

    # Collect per-head norm across all prompts
    # Shape: [n_prompts, n_layers, n_heads]
    prompt_names = list(all_results.keys())
    layer_indices = sorted(next(iter(all_results.values())).keys())

    norm_tensor = []
    for pname in prompt_names:
        layer_norms = []
        for lidx in layer_indices:
            layer_norms.append(all_results[pname][lidx]["norms"])
        norm_tensor.append(torch.stack(layer_norms))

    # [n_prompts, n_layers, n_heads]
    norm_tensor = torch.stack(norm_tensor)

    # Variance across prompts per head: [n_layers, n_heads]
    variance = norm_tensor.var(dim=0)

    # Mean across prompts: [n_layers, n_heads]
    mean_norms = norm_tensor.mean(dim=0)

    # Coefficient of variation (normalized variance)
    cv = variance.sqrt() / (mean_norms + 1e-8)

    # ── Top 10 most task-sensitive heads (highest CV) ──
    print(f"\n  Top 15 TASK-SENSITIVE heads (highest variance across prompts):")
    print(f"  {'Rank':>4s}  {'Layer':>5s}  {'Head':>4s}  {'CV':>8s}  {'Mean Norm':>10s}  {'Var':>8s}")
    print(f"  {'─' * 50}")

    flat_cv = cv.flatten()
    top_k = min(15, flat_cv.numel())
    top_vals, top_indices = flat_cv.topk(top_k)

    sensitive_heads = []
    for rank, (val, idx) in enumerate(zip(top_vals, top_indices)):
        layer_pos = idx.item() // num_heads
        head_pos = idx.item() % num_heads
        layer_id = layer_indices[layer_pos]
        mn = mean_norms[layer_pos, head_pos].item()
        vr = variance[layer_pos, head_pos].item()
        print(f"  {rank+1:4d}  L{layer_id:3d}  H{head_pos:3d}  {val.item():8.4f}  {mn:10.4f}  {vr:8.4f}")
        sensitive_heads.append((layer_id, head_pos))

    # ── Top 10 most stable heads (lowest CV with meaningful norm) ──
    print(f"\n  Top 15 INFRASTRUCTURE heads (lowest variance, significant norm):")
    print(f"  {'Rank':>4s}  {'Layer':>5s}  {'Head':>4s}  {'CV':>8s}  {'Mean Norm':>10s}")
    print(f"  {'─' * 50}")

    # Filter out near-zero heads
    significant = mean_norms > mean_norms.mean() * 0.3
    stable_cv = cv.clone()
    stable_cv[~significant] = float('inf')

    flat_stable = stable_cv.flatten()
    bottom_vals, bottom_indices = flat_stable.topk(min(15, flat_stable.numel()), largest=False)

    stable_heads = []
    for rank, (val, idx) in enumerate(zip(bottom_vals, bottom_indices)):
        if val.item() == float('inf'):
            break
        layer_pos = idx.item() // num_heads
        head_pos = idx.item() % num_heads
        layer_id = layer_indices[layer_pos]
        mn = mean_norms[layer_pos, head_pos].item()
        print(f"  {rank+1:4d}  L{layer_id:3d}  H{head_pos:3d}  {val.item():8.4f}  {mn:10.4f}")
        stable_heads.append((layer_id, head_pos))

    # ── Per-prompt breakdown for top sensitive heads ──
    print(f"\n  Task-sensitive heads — per-prompt norms:")
    print(f"  {'Head':>10s}", end="")
    for pname in prompt_names:
        print(f"  {pname:>10s}", end="")
    print()
    print(f"  {'─' * (12 + 12 * len(prompt_names))}")

    for layer_id, head_pos in sensitive_heads[:10]:
        layer_pos = layer_indices.index(layer_id)
        print(f"  L{layer_id:2d}:H{head_pos:02d}  ", end="")
        for pi, pname in enumerate(prompt_names):
            val = norm_tensor[pi, layer_pos, head_pos].item()
            print(f"  {val:10.4f}", end="")
        print()

    # ── Gate analysis if available ──
    has_gates = any(
        all_results[pname][layer_indices[0]].get("gate") is not None
        for pname in prompt_names
    )
    if has_gates:
        print(f"\n  Gate activation (sigmoid) — mean across prompts:")
        print(f"  {'':>6s}", end="")
        for h in range(num_heads):
            print(f"  H{h:02d}", end="")
        print()

        for lidx in layer_indices:
            gates = []
            for pname in prompt_names:
                g = all_results[pname][lidx].get("gate")
                if g is not None:
                    gates.append(g)
            if gates:
                mean_gate = torch.stack(gates).mean(dim=0)
                print(f"  L{lidx:2d}:  ", end="")
                for h in range(num_heads):
                    v = mean_gate[h].item()
                    if v > 0.7:
                        marker = "█"
                    elif v > 0.5:
                        marker = "▓"
                    elif v > 0.3:
                        marker = "▒"
                    else:
                        marker = "░"
                    print(f"  {marker} ", end="")
                print(f"  [{mean_gate.min().item():.2f}-{mean_gate.max().item():.2f}]")

    return {
        "sensitive_heads": sensitive_heads,
        "stable_heads": stable_heads,
        "norm_tensor": norm_tensor,
        "variance": variance,
        "cv": cv,
        "mean_norms": mean_norms,
        "layer_indices": layer_indices,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-head attention explorer")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Tokens to generate per prompt (keep low for speed)")
    parser.add_argument("--output-dir", default="data/per_head")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"\n  Per-Head Attention Explorer")
    print(f"  Model:      {args.model}")
    print(f"  Device:     {args.device}")
    print(f"  Dtype:      {args.dtype}")
    print(f"  Max tokens: {args.max_tokens}\n")

    # Load model
    model = load_model(args.model, args.device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Get num_heads from config
    config = model.config
    text_cfg = getattr(config, "text_config", config)
    num_heads = text_cfg.num_attention_heads

    print(f"  Architecture: {num_heads} heads, head_dim={text_cfg.hidden_size // num_heads}")
    print(f"  Full attention layers: {QWEN35_FA_LAYERS}\n")

    # Install per-head capture
    capture = PerHeadCapture()
    capture.install(model)

    # Run each prompt
    all_results = {}

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"  Running: {prompt_name}")
        print(f"    Prompt: {prompt_text[:60]}...")

        capture.reset()

        # Tokenize
        full_prompt = f"Question: {prompt_text}\nAnswer:"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(args.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,  # greedy for reproducibility
                temperature=1.0,
            )

        n_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        print(f"    Generated {n_tokens} tokens")

        # Aggregate
        results = capture.aggregate()
        all_results[prompt_name] = results

        # Print per-prompt heatmap
        norms_data = {k: v["norms"] for k, v in results.items()}
        print_heatmap(
            f"Per-Head Norms — {prompt_name}",
            norms_data, num_heads, label="norm"
        )

        if any(v.get("entropy") is not None for v in results.values()):
            entropy_data = {k: v["entropy"] for k, v in results.items()
                          if v.get("entropy") is not None}
            if entropy_data:
                print_heatmap(
                    f"Per-Head Attention Entropy — {prompt_name}",
                    entropy_data, num_heads, label="bits"
                )

    # Cross-prompt analysis
    analysis = print_analysis(all_results, num_heads)

    # Save raw data
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "per_head_exploration.json")

    save_data = {
        "model": args.model,
        "num_heads": num_heads,
        "fa_layers": QWEN35_FA_LAYERS,
        "prompts": PROMPTS,
    }

    for pname, results in all_results.items():
        save_data[pname] = {}
        for lidx, data in results.items():
            save_data[pname][str(lidx)] = {
                "norms": data["norms"].tolist(),
            }
            if data.get("entropy") is not None:
                save_data[pname][str(lidx)]["entropy"] = data["entropy"].tolist()
            if data.get("gate") is not None:
                save_data[pname][str(lidx)]["gate"] = data["gate"].tolist()

    # Save analysis
    save_data["analysis"] = {
        "sensitive_heads": analysis["sensitive_heads"],
        "stable_heads": analysis["stable_heads"],
        "variance": {str(analysis["layer_indices"][i]): analysis["variance"][i].tolist()
                     for i in range(len(analysis["layer_indices"]))},
        "cv": {str(analysis["layer_indices"][i]): analysis["cv"][i].tolist()
               for i in range(len(analysis["layer_indices"]))},
        "mean_norms": {str(analysis["layer_indices"][i]): analysis["mean_norms"][i].tolist()
                       for i in range(len(analysis["layer_indices"]))},
    }

    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {save_path}")
    print(f"\n  Done!")

    # Cleanup
    capture.cleanup()


if __name__ == "__main__":
    main()
