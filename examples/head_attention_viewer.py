"""Visualise what specific attention heads are looking at.

Shows the actual attention weight patterns for selected heads — which tokens
attend to which. Runs a single prompt and prints attention maps.

Usage:
    python examples/head_attention_viewer.py
    python examples/head_attention_viewer.py --prompt "your question here"
"""

import argparse
import json
import gc
import torch
from collections import defaultdict
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


# ── Attention pattern capture ─────────────────────────────────────

QWEN35_FA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]

# Heads we want to inspect (from explorer results)
INTERESTING_HEADS = [
    (3, 11, "L3:H11 — task-sensitive, top in L3"),
    (3, 3,  "L3:H03 — task-sensitive"),
    (7, 5,  "L7:H05 — #1 most task-sensitive, creativity head"),
    (7, 4,  "L7:H04 — task-sensitive"),
    (11, 14, "L11:H14 — task-sensitive"),
    (15, 9,  "L15:H09 — task-sensitive"),
    (31, 0,  "L31:H00 — precision/convergence head"),
    (19, 3,  "L19:H03 — infrastructure (stable)"),
    (23, 3,  "L23:H03 — infrastructure (stable)"),
    (27, 3,  "L27:H03 — infrastructure (stable)"),
]


class AttentionCapture:
    """Captures full attention weight matrices for selected heads."""

    def __init__(self):
        self.hooks = []
        # Store attention weights per layer: {layer_idx: [batch, heads, seq_q, seq_k]}
        self.attn_weights = {}
        # Store gate values per layer
        self.gate_vals = {}

    def install(self, model):
        from dflux.multiscale_telemetry import MultiScaleTelemetry
        layers = MultiScaleTelemetry._find_transformer_layers(model)

        target_layers = set(l for l, h, _ in INTERESTING_HEADS)

        for layer_idx in target_layers:
            if layer_idx >= len(layers):
                continue

            layer = layers[layer_idx]
            attn = MultiScaleTelemetry._find_attn_module(layer)
            if attn is None or not hasattr(attn, 'o_proj'):
                continue

            capture = self
            lidx = layer_idx

            def make_hooked_forward(cap, li, attn_mod):
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

                    # Force eager attention to get weights back
                    # (SDPA/flash don't return attention weights)
                    attn_output, attn_weights = eager_attention_forward(
                        attn_mod,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        dropout=0.0 if not attn_mod.training else attn_mod.attention_dropout,
                        scaling=attn_mod.scaling,
                        **kwargs,
                    )

                    # ══ CAPTURE attention weights ══
                    with torch.no_grad():
                        if attn_weights is not None:
                            # Only keep the prefill pass (full seq x seq matrix)
                            # During generation, attn_weights is [batch, heads, 1, seq]
                            # We want the first call which has the full prompt
                            if li not in cap.attn_weights:
                                cap.attn_weights[li] = attn_weights.float().cpu()

                            # Capture gate values
                            gate_sigmoid = torch.sigmoid(gate_raw.float())
                            if li not in cap.gate_vals:
                                cap.gate_vals[li] = gate_sigmoid.cpu()

                    # Normal forward continues
                    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                    attn_output = attn_output * torch.sigmoid(gate_raw)
                    attn_output = attn_mod.o_proj(attn_output)

                    return attn_output, attn_weights

                return hooked_forward

            attn.forward = make_hooked_forward(capture, lidx, attn)
            self.hooks.append((attn, attn.forward))
            print(f"    Hooked L{layer_idx} [FA]")

    def cleanup(self):
        # Note: we overwrote forward, harder to restore.
        # For a one-shot script this is fine.
        pass


# ── Visualisation ─────────────────────────────────────────────────

def print_attention_map(head_label: str, weights: torch.Tensor, tokens: list,
                        max_display: int = 50):
    """Print attention pattern for one head.

    weights: [seq_q, seq_k] — attention from each query to each key
    tokens: list of token strings
    """
    seq_len = min(weights.shape[0], max_display)

    print(f"\n{'─' * 72}")
    print(f"  {head_label}")
    print(f"  Pattern: what each token attends to (top-3 targets)")
    print(f"{'─' * 72}")

    for q in range(seq_len):
        q_tok = tokens[q][:12].ljust(12)
        attn_row = weights[q, :q+1]  # causal: can only attend to past + self

        # Top 3 attended positions
        k = min(3, attn_row.shape[0])
        top_vals, top_idxs = attn_row.topk(k)

        targets = []
        for val, idx in zip(top_vals, top_idxs):
            if val.item() > 0.01:  # skip negligible
                k_tok = tokens[idx.item()][:10]
                targets.append(f"{k_tok}({val.item():.2f})")

        targets_str = "  ".join(targets)
        print(f"  {q:3d} [{q_tok}] → {targets_str}")


def print_attention_summary(head_label: str, weights: torch.Tensor, tokens: list):
    """Print summary statistics for a head's attention pattern."""
    seq_len = weights.shape[0]

    print(f"\n{'=' * 72}")
    print(f"  {head_label}")
    print(f"{'=' * 72}")

    # 1. Self-attention ratio: how much does each token attend to itself?
    diag = torch.diag(weights[:seq_len, :seq_len])
    self_attn_mean = diag.mean().item()
    self_attn_std = diag.std().item()

    # 2. Recency bias: average attention to last 5 tokens vs rest
    recency_scores = []
    for q in range(5, seq_len):
        recent = weights[q, max(0, q-5):q].sum().item()
        total = weights[q, :q].sum().item()
        if total > 0:
            recency_scores.append(recent / total)
    recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0

    # 3. BOS/first token attention
    bos_attn = weights[:, 0].mean().item()

    # 4. Entropy of attention distribution (mean across queries)
    ent = -(weights.clamp(min=1e-10) * weights.clamp(min=1e-10).log()).sum(dim=-1)
    mean_entropy = ent.mean().item()
    max_possible_entropy = torch.log(torch.arange(1, seq_len + 1, dtype=torch.float32))
    normalized_entropy = (ent / max_possible_entropy.clamp(min=1e-10)).mean().item()

    # 5. Positional pattern: does it attend to specific relative positions?
    # Average attention by relative position (how far back)
    max_dist = min(20, seq_len)
    pos_profile = torch.zeros(max_dist)
    counts = torch.zeros(max_dist)
    for q in range(seq_len):
        for offset in range(min(max_dist, q + 1)):
            k_pos = q - offset
            pos_profile[offset] += weights[q, k_pos].item()
            counts[offset] += 1
    pos_profile = pos_profile / counts.clamp(min=1)

    # 6. Token type attention: find tokens that get disproportionate attention
    mean_received = weights.sum(dim=0) / seq_len  # mean attention received per key
    top_received_vals, top_received_idxs = mean_received.topk(min(5, seq_len))

    print(f"  Self-attention:     {self_attn_mean:.3f} (std {self_attn_std:.3f})")
    print(f"  BOS/first attention: {bos_attn:.3f}")
    print(f"  Recency bias:       {recency:.3f} (fraction of attn to last 5 tokens)")
    print(f"  Mean entropy:        {mean_entropy:.2f} (normalized: {normalized_entropy:.2f})")

    # Classify head type
    head_type = []
    if self_attn_mean > 0.3:
        head_type.append("SELF-REFERENTIAL")
    if bos_attn > 0.2:
        head_type.append("BOS-ANCHOR")
    if recency > 0.7:
        head_type.append("LOCAL/RECENT")
    if normalized_entropy > 0.7:
        head_type.append("DIFFUSE/GLOBAL")
    if normalized_entropy < 0.3:
        head_type.append("FOCUSED/SPARSE")

    if not head_type:
        head_type.append("MIXED")

    print(f"  Head type:           {', '.join(head_type)}")

    # Positional profile
    print(f"\n  Positional profile (avg attn by distance back):")
    print(f"    {'Offset':>6s}  {'Attn':>8s}  Bar")
    bar_max = pos_profile.max().item()
    for offset in range(max_dist):
        v = pos_profile[offset].item()
        bar_len = int(40 * v / bar_max) if bar_max > 0 else 0
        label = "self" if offset == 0 else f"-{offset}"
        print(f"    {label:>6s}  {v:8.4f}  {'█' * bar_len}")

    # Most attended tokens
    print(f"\n  Most attended tokens (highest mean incoming attention):")
    for val, idx in zip(top_received_vals, top_received_idxs):
        tok = tokens[idx.item()] if idx.item() < len(tokens) else "?"
        print(f"    pos {idx.item():3d} [{tok:>15s}]  attn={val.item():.4f}")


def print_gate_analysis(gate_vals: dict, tokens: list, num_heads: int):
    """Show gate activation patterns."""
    print(f"\n{'=' * 72}")
    print(f"  GATE VALUES (which heads the model chooses to listen to)")
    print(f"{'=' * 72}")

    for layer_idx in sorted(gate_vals.keys()):
        gate = gate_vals[layer_idx][0]  # [seq, num_heads]
        # Mean gate across sequence
        mean_gate = gate.mean(dim=0)  # [num_heads]

        print(f"\n  L{layer_idx} mean gate per head:")
        for h in range(num_heads):
            v = mean_gate[h].item()
            bar_len = int(30 * v)
            status = "OPEN" if v > 0.6 else "partial" if v > 0.3 else "CLOSED"
            print(f"    H{h:02d}: {v:.3f} {'█' * bar_len} {status}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Head attention pattern viewer")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt (default: cucumber question)")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    prompt_text = args.prompt or (
        "What is the temperature on the far side of the moon, "
        "and what would happen to a cucumber if it suddenly appeared there."
    )

    print(f"\n  Head Attention Viewer")
    print(f"  Model:  {args.model}")
    print(f"  Prompt: {prompt_text[:60]}...\n")

    # Load model
    model = load_model(args.model, args.device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    text_cfg = getattr(model.config, "text_config", model.config)
    num_heads = text_cfg.num_attention_heads

    # Install capture hooks
    capture = AttentionCapture()
    capture.install(model)

    # Tokenize and run prefill only (no generation needed for attention patterns)
    full_prompt = f"Question: {prompt_text}\nAnswer:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(args.device)
    tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    print(f"  Tokens: {len(tokens)}")
    print(f"  Prompt tokens: {' '.join(t[:8] for t in tokens[:20])}{'...' if len(tokens) > 20 else ''}")
    print()

    # Single forward pass to capture attention weights
    with torch.no_grad():
        model(**inputs)

    # Print results for each interesting head
    for layer_idx, head_idx, label in INTERESTING_HEADS:
        if layer_idx not in capture.attn_weights:
            print(f"  [SKIP] {label} — no weights captured")
            continue

        weights = capture.attn_weights[layer_idx]
        # weights shape: [batch, num_heads, seq_q, seq_k]
        head_weights = weights[0, head_idx]  # [seq_q, seq_k]

        print_attention_summary(label, head_weights, tokens)
        print_attention_map(f"{label} — token-level detail", head_weights, tokens,
                           max_display=min(30, len(tokens)))

    # Gate analysis
    print_gate_analysis(capture.gate_vals, tokens, num_heads)

    # Save raw data
    import os
    os.makedirs("data/per_head", exist_ok=True)
    save_path = "data/per_head/attention_patterns.json"

    save_data = {
        "model": args.model,
        "prompt": prompt_text,
        "tokens": tokens,
        "heads": [],
    }

    for layer_idx, head_idx, label in INTERESTING_HEADS:
        if layer_idx not in capture.attn_weights:
            continue
        weights = capture.attn_weights[layer_idx][0, head_idx]
        save_data["heads"].append({
            "layer": layer_idx,
            "head": head_idx,
            "label": label,
            "weights_shape": list(weights.shape),
            # Save just the summary stats, not the full matrix
            "self_attn_mean": torch.diag(weights).mean().item(),
            "bos_attn": weights[:, 0].mean().item(),
            "mean_entropy": (-(weights.clamp(min=1e-10) * weights.clamp(min=1e-10).log()).sum(dim=-1)).mean().item(),
        })

    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Saved to {save_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
