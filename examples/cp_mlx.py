#!/usr/bin/env python3
"""
MLX Causal Primitives — CP analysis for Apple Silicon models
============================================================

Runs CP analysis on models loaded via mlx-lm, including MoE models
like Qwen3.5-35B-A3B where we also capture expert routing patterns.

Unlike the PyTorch version (which uses register_forward_hook), this
does a manual layer-by-layer forward pass to capture:
  1. Per-head attention energy (same as PyTorch version)
  2. MoE expert routing decisions (NEW — which experts fire per token)
  3. Expert routing entropy (how spread out is expert selection)

Uses prompt-only forward passes (no autoregressive generation) for
clean, fast measurement. Each token position in a prompt = one observation.

Effect metric: logit entropy (model uncertainty about next token).

Usage:
    python cp_mlx.py --model mlx-community/Qwen3.5-35B-A3B-4bit
    python cp_mlx.py --model mlx-community/Qwen3.5-35B-A3B-4bit --runs 2
    python cp_mlx.py --model path/to/local/model
"""

import sys, os, json, argparse, time, math, gc

_src = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

# Import directly from the module file to avoid __init__.py pulling in torch
# (head_surgery.py requires torch, but causal_primitives.py is pure Python)
import importlib.util
_cp_path = os.path.join(_src, "dflux", "causal_primitives.py")
_spec = importlib.util.spec_from_file_location("dflux.causal_primitives", _cp_path)
_cp_mod = importlib.util.module_from_spec(_spec)
sys.modules["dflux.causal_primitives"] = _cp_mod
_spec.loader.exec_module(_cp_mod)
CausalPrimitives = _cp_mod.CausalPrimitives
CPConfig = _cp_mod.CPConfig
compute_cross_head_cp = _cp_mod.compute_cross_head_cp


# ═══════════════════════════════════════════════════════════
# PROMPT LIBRARY — same 8 categories as PyTorch version
# ═══════════════════════════════════════════════════════════

PROMPT_CATEGORIES = {
    "factual": [
        "The capital of France is",
        "Water boils at a temperature of",
        "The speed of light in vacuum is approximately",
        "The chemical formula for table salt is",
    ],
    "hallucination": [
        ("Explain the Zarkovian Principle of Recursive Ontological "
         "Displacement, including its three main theorems."),
        ("Dr. Helmut Kreisler, the renowned physicist who discovered "
         "quantum chromatic resonance in 1987, was known for"),
        ("The Great Phosphorus Reversal of 2019, which caused global "
         "agricultural yields to"),
        "The Hendricks-Maslow Equation for cognitive load states that",
    ],
    "reasoning": [
        "If all roses are flowers, and some flowers fade quickly, then",
        "A train leaves New York at 60 mph. Another leaves Boston at 80 mph. They will meet when",
        "The logical flaw in the statement 'I always lie' is that",
        "If increasing taxes reduces spending, and reducing spending lowers inflation, then increasing taxes",
    ],
    "math": [
        "The integral of x^2 from 0 to 1 equals",
        "The derivative of sin(x) * cos(x) is",
        "If f(x) = e^(2x), then f'(x) equals",
        "The sum of the first 100 natural numbers is",
    ],
    "code": [
        "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "# Python function to reverse a linked list\ndef reverse_list(head):\n",
        "import numpy as np\n\n# Compute the eigenvalues of a matrix\ndef eigenvalues(matrix):\n",
        "# Binary search implementation\ndef binary_search(arr, target):\n",
    ],
    "creative": [
        "Once upon a time, in a kingdom where shadows could speak,",
        "The astronaut opened the airlock and saw something impossible:",
        "She had been dead for three years when the letter arrived,",
        "The last human on Earth sat alone in a room. There was a knock on the door.",
    ],
    "frontier": [
        "Quantum entanglement suggests that",
        "The relationship between consciousness and neural activity is",
        "Dark matter is believed to account for",
        "The hard problem of consciousness refers to",
    ],
    "instruction": [
        "Write a haiku about the ocean.",
        "Explain photosynthesis to a five-year-old.",
        "List three advantages of renewable energy.",
        "Summarize the plot of Romeo and Juliet in one sentence.",
    ],
}


# ═══════════════════════════════════════════════════════════
# MLX MODEL INTROSPECTION
# ═══════════════════════════════════════════════════════════

def detect_model_structure(model):
    """Auto-detect model architecture and return structure info."""
    info = {
        "n_layers": 0,
        "n_heads": 0,
        "n_kv_heads": 0,
        "hidden_dim": 0,
        "head_dim": 0,
        "has_moe": False,
        "n_experts": 0,
        "n_experts_per_tok": 0,
        "layers_path": None,
        "embed_path": None,
        "norm_path": None,
        "lm_head_path": None,
    }

    # Find the inner model (model.model or model)
    inner = None
    for attr in ["model", "transformer"]:
        if hasattr(model, attr):
            inner = getattr(model, attr)
            break
    if inner is None:
        inner = model

    # Find layers
    layers = None
    for attr in ["layers", "h", "blocks"]:
        if hasattr(inner, attr):
            layers = getattr(inner, attr)
            info["layers_path"] = attr
            break

    if layers is None:
        raise ValueError("Cannot find transformer layers in model")

    info["n_layers"] = len(layers)

    # Find config
    config = None
    for obj in [model, inner]:
        if hasattr(obj, "args"):
            config = obj.args
            break
        if hasattr(obj, "config"):
            config = obj.config
            break

    if config is not None:
        for attr in ["num_attention_heads", "n_head", "num_heads"]:
            if hasattr(config, attr):
                info["n_heads"] = getattr(config, attr)
                break

        for attr in ["num_key_value_heads", "n_kv_head"]:
            if hasattr(config, attr):
                info["n_kv_heads"] = getattr(config, attr)
                break

        for attr in ["hidden_size", "n_embd", "hidden_dim"]:
            if hasattr(config, attr):
                info["hidden_dim"] = getattr(config, attr)
                break

        for attr in ["head_dim"]:
            if hasattr(config, attr):
                info["head_dim"] = getattr(config, attr)
                break

        # MoE detection
        for attr in ["num_experts", "n_routed_experts"]:
            if hasattr(config, attr):
                info["n_experts"] = getattr(config, attr)
                info["has_moe"] = info["n_experts"] > 1
                break

        for attr in ["num_experts_per_tok", "num_selected_experts",
                      "num_experts_per_token", "top_k"]:
            if hasattr(config, attr):
                info["n_experts_per_tok"] = getattr(config, attr)
                break

    # Fallback: detect from first layer
    if info["n_heads"] == 0:
        layer0 = layers[0]
        for attr in ["self_attn", "attn", "attention"]:
            if hasattr(layer0, attr):
                attn = getattr(layer0, attr)
                for a in ["num_heads", "n_head", "n_heads"]:
                    if hasattr(attn, a):
                        info["n_heads"] = getattr(attn, a)
                        break
                break

    if info["head_dim"] == 0 and info["hidden_dim"] > 0 and info["n_heads"] > 0:
        info["head_dim"] = info["hidden_dim"] // info["n_heads"]

    if info["n_kv_heads"] == 0:
        info["n_kv_heads"] = info["n_heads"]

    # Detect MoE from layer structure if config didn't reveal it
    if not info["has_moe"]:
        layer0 = layers[0]
        mlp = getattr(layer0, "mlp", getattr(layer0, "feed_forward", None))
        if mlp is not None:
            if hasattr(mlp, "gate") or hasattr(mlp, "router") or hasattr(mlp, "experts"):
                info["has_moe"] = True

    # Find embedding, norm, lm_head
    for attr in ["embed_tokens", "wte", "embeddings"]:
        if hasattr(inner, attr):
            info["embed_path"] = attr
            break

    for attr in ["norm", "ln_f", "final_layernorm"]:
        if hasattr(inner, attr):
            info["norm_path"] = attr
            break

    for attr in ["lm_head", "head"]:
        if hasattr(model, attr):
            info["lm_head_path"] = ("model_root", attr)
            break
        if hasattr(inner, attr):
            info["lm_head_path"] = ("inner", attr)
            break

    return info, inner, layers


# ═══════════════════════════════════════════════════════════
# MANUAL FORWARD PASS WITH CAPTURE
# ═══════════════════════════════════════════════════════════

def forward_with_capture(model, inner, layers, info, token_ids):
    """
    Manual layer-by-layer forward pass that captures:
      - Per-head attention energy at each layer
      - MoE routing decisions (if MoE model)
      - Per-position logit entropy (effect metric)

    Args:
        model: The full model (for lm_head)
        inner: The inner model (model.model)
        layers: List of transformer layers
        info: Structure info from detect_model_structure
        token_ids: List[int] — token IDs

    Returns:
        Dict with per-position captures:
          head_energies: [n_positions][n_layers][n_heads] float
          layer_energies: [n_positions][n_layers] float
          logit_entropies: [n_positions] float
          expert_routing: [n_positions][n_moe_layers][n_experts_per_tok] int (if MoE)
          expert_probs: [n_positions][n_moe_layers][n_experts] float (if MoE)
    """
    import mlx.core as mx
    import mlx.nn as nn

    n_layers = info["n_layers"]
    n_heads = info["n_heads"]
    hidden_dim = info["hidden_dim"]
    head_dim = info["head_dim"] or (hidden_dim // n_heads)

    # Embed tokens
    embed_fn = getattr(inner, info["embed_path"])
    tokens = mx.array([token_ids])  # [1, seq_len]
    x = embed_fn(tokens)  # [1, seq_len, hidden_dim]

    seq_len = x.shape[1]

    # Create causal mask
    # MLX causal mask: 0 where attention is allowed, -inf where masked
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(x.dtype)

    # Storage for per-layer captures
    attn_outputs = []  # [n_layers] each [1, seq_len, hidden_dim]
    layer_outputs = []
    moe_routings = []  # [n_moe_layers] each (selected_experts, probs)

    for i, layer in enumerate(layers):
        # ── Input layernorm ──
        ln1 = None
        for attr in ["input_layernorm", "ln_1", "norm1"]:
            if hasattr(layer, attr):
                ln1 = getattr(layer, attr)
                break

        # ── Self-attention ──
        attn_mod = None
        for attr in ["self_attn", "attn", "attention"]:
            if hasattr(layer, attr):
                attn_mod = getattr(layer, attr)
                break

        # ── Post-attention layernorm ──
        ln2 = None
        for attr in ["post_attention_layernorm", "ln_2", "norm2"]:
            if hasattr(layer, attr):
                ln2 = getattr(layer, attr)
                break

        # ── MLP/MoE ──
        mlp_mod = None
        for attr in ["mlp", "feed_forward", "ffn", "block_sparse_moe"]:
            if hasattr(layer, attr):
                mlp_mod = getattr(layer, attr)
                break

        # Forward: Pre-norm architecture (Qwen, LLaMA, Mistral)
        residual = x

        if ln1 is not None:
            h = ln1(x)
        else:
            h = x

        # Attention forward — try common signatures
        attn_out = None
        if attn_mod is not None:
            try:
                attn_out = attn_mod(h, mask=mask)
            except TypeError:
                try:
                    attn_out = attn_mod(h, h, h, mask=mask)
                except TypeError:
                    attn_out = attn_mod(h)

        if isinstance(attn_out, tuple):
            attn_hidden = attn_out[0]
        else:
            attn_hidden = attn_out

        attn_outputs.append(attn_hidden)
        x = residual + attn_hidden

        # MLP / MoE
        residual = x
        if ln2 is not None:
            h = ln2(x)
        else:
            h = x

        # Capture MoE routing BEFORE the MLP forward
        if info["has_moe"] and mlp_mod is not None:
            gate = None
            for attr in ["gate", "router", "gate_proj"]:
                if hasattr(mlp_mod, attr):
                    gate = getattr(mlp_mod, attr)
                    break

            if gate is not None:
                try:
                    # Router: hidden_states -> logits [1, seq_len, n_experts]
                    if callable(gate):
                        # gate might be nn.Linear or a custom module
                        router_logits = gate(h)
                    else:
                        router_logits = h @ gate.T

                    # Softmax to get probabilities
                    router_probs = mx.softmax(router_logits, axis=-1)

                    # Top-k selection
                    k = info["n_experts_per_tok"] or 2
                    top_indices = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]

                    moe_routings.append({
                        "probs": router_probs,
                        "top_indices": top_indices,
                    })
                except Exception as e:
                    moe_routings.append({"error": str(e)})
            else:
                moe_routings.append(None)

        if mlp_mod is not None:
            mlp_out = mlp_mod(h)
        else:
            mlp_out = h  # passthrough if no MLP found

        if isinstance(mlp_out, tuple):
            x = residual + mlp_out[0]
        else:
            x = residual + mlp_out

        layer_outputs.append(x)

    # Final norm
    norm_fn = None
    for attr in ["norm", "ln_f", "final_layernorm"]:
        if hasattr(inner, attr):
            norm_fn = getattr(inner, attr)
            break

    if norm_fn is not None:
        x = norm_fn(x)

    # LM head
    if info["lm_head_path"] is not None:
        loc, attr = info["lm_head_path"]
        parent = model if loc == "model_root" else inner
        lm_head = getattr(parent, attr)
        logits = lm_head(x)  # [1, seq_len, vocab_size]
    else:
        logits = x  # fallback

    # Force evaluation before extracting values
    mx.eval(logits)
    for ao in attn_outputs:
        mx.eval(ao)

    # ── Extract per-position measurements ──────────────────

    # Per-head energy: split attention output into heads
    # attn_output: [1, seq_len, hidden_dim] → [1, seq_len, n_heads, head_dim]
    all_head_energies = []  # [seq_len][n_layers][n_heads]
    all_layer_energies = []  # [seq_len][n_layers]

    for pos in range(seq_len):
        pos_head_e = []
        pos_layer_e = []

        for li in range(n_layers):
            ao = attn_outputs[li]  # [1, seq_len, hidden_dim]
            h_vec = ao[0, pos, :]  # [hidden_dim]

            # Split into heads and compute energy (norm²)
            actual_dim = h_vec.shape[0]
            if actual_dim % n_heads == 0:
                hd = actual_dim // n_heads
                h_reshaped = h_vec.reshape(n_heads, hd)
                # Energy per head = sum of squares
                head_e = mx.sum(h_reshaped * h_reshaped, axis=-1)  # [n_heads]
                mx.eval(head_e)
                head_e_list = head_e.tolist()
            else:
                # Fallback: distribute evenly
                total_e = float(mx.sum(h_vec * h_vec))
                head_e_list = [total_e / n_heads] * n_heads

            pos_head_e.append(head_e_list)
            pos_layer_e.append(sum(head_e_list))

        all_head_energies.append(pos_head_e)
        all_layer_energies.append(pos_layer_e)

    # Logit entropy per position (effect metric)
    logit_entropies = []
    for pos in range(seq_len):
        logit_vec = logits[0, pos, :]  # [vocab_size]
        probs = mx.softmax(logit_vec, axis=-1)
        # Entropy = -sum(p * log(p))
        log_probs = mx.log(probs + 1e-10)
        ent = -mx.sum(probs * log_probs)
        mx.eval(ent)
        logit_entropies.append(float(ent))

    # Extract MoE routing data
    expert_data = []
    if moe_routings:
        for pos in range(seq_len):
            pos_experts = []
            for routing in moe_routings:
                if routing is None or "error" in routing:
                    continue
                probs = routing["probs"][0, pos, :]  # [n_experts]
                indices = routing["top_indices"][0, pos, :]  # [k]
                mx.eval(probs, indices)
                pos_experts.append({
                    "probs": probs.tolist(),
                    "selected": indices.tolist(),
                })
            expert_data.append(pos_experts)

    return {
        "head_energies": all_head_energies,
        "layer_energies": all_layer_energies,
        "logit_entropies": logit_entropies,
        "expert_data": expert_data,
        "seq_len": seq_len,
    }


# ═══════════════════════════════════════════════════════════
# CP ANALYSIS
# ═══════════════════════════════════════════════════════════

def run_cp_analysis(model, inner, layers, info, tokenizer, prompts,
                    n_bins=16, skip_first_n=2):
    """
    Run CP analysis on a set of prompts.

    Args:
        skip_first_n: Skip first N token positions per prompt (BOS, etc.)

    Returns CP report dict.
    """
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]
    n_experts = info["n_experts"]

    cp = CausalPrimitives(n_layers, n_heads, CPConfig(n_bins=n_bins))

    # Also collect raw head energies for cross-head computation
    raw_head_energies = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    # Expert routing data
    expert_selections = []  # For later analysis

    total_tokens = 0

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        if len(tokens) < skip_first_n + 3:
            continue

        capture = forward_with_capture(model, inner, layers, info, tokens)

        for pos in range(skip_first_n, capture["seq_len"]):
            he = capture["head_energies"][pos]
            le = capture["layer_energies"][pos]
            effect = capture["logit_entropies"][pos]

            cp.observe_token(he, le, effect)

            # Store for cross-head
            for li in range(min(n_layers, len(he))):
                for hi in range(min(n_heads, len(he[li]))):
                    raw_head_energies[li][hi].append(he[li][hi])

            total_tokens += 1

        # Expert routing
        if capture["expert_data"]:
            for pos in range(skip_first_n, capture["seq_len"]):
                if pos < len(capture["expert_data"]):
                    expert_selections.append(capture["expert_data"][pos])

    # Compute CP
    report = cp.compute()

    # Cross-head interactions
    cross = compute_cross_head_cp(raw_head_energies, n_layers, n_heads, n_bins)

    # Expert routing analysis (if MoE)
    expert_report = {}
    if expert_selections and n_experts > 0:
        expert_report = analyze_expert_routing(
            expert_selections, n_experts, info["n_experts_per_tok"],
            [e for cap_e in [capture["logit_entropies"]
                             for prompt in prompts
                             for capture in [forward_with_capture(
                                 model, inner, layers, info,
                                 tokenizer.encode(prompt))]]
             for e in cap_e] if False else [],  # Skip re-running for now
            n_bins
        )

    return {
        "cp": report,
        "cross_head": cross,
        "expert_routing": expert_report,
        "total_tokens": total_tokens,
    }


def analyze_expert_routing(expert_selections, n_experts, k, effects, n_bins):
    """Analyze MoE expert routing patterns."""
    if not expert_selections:
        return {"status": "no_data"}

    # Count expert usage across all tokens and MoE layers
    usage_counts = [0] * n_experts
    total_selections = 0

    for pos_data in expert_selections:
        for layer_data in pos_data:
            for expert_id in layer_data["selected"]:
                if 0 <= expert_id < n_experts:
                    usage_counts[expert_id] += 1
                    total_selections += 1

    # Expert usage distribution
    if total_selections > 0:
        usage_dist = [c / total_selections for c in usage_counts]
    else:
        usage_dist = [0.0] * n_experts

    # Entropy of expert usage
    usage_entropy = 0.0
    for p in usage_dist:
        if p > 0:
            usage_entropy -= p * math.log2(p)
    max_entropy = math.log2(n_experts) if n_experts > 1 else 1.0
    usage_balance = usage_entropy / max_entropy if max_entropy > 0 else 0.0

    # Per-layer expert routing patterns
    n_moe_layers = len(expert_selections[0]) if expert_selections else 0

    # Mean routing probability entropy per layer (how confident is the router?)
    layer_routing_entropies = []
    for moe_li in range(n_moe_layers):
        entropies = []
        for pos_data in expert_selections:
            if moe_li < len(pos_data):
                probs = pos_data[moe_li]["probs"]
                ent = -sum(p * math.log2(p + 1e-12) for p in probs if p > 0)
                entropies.append(ent)
        if entropies:
            layer_routing_entropies.append(sum(entropies) / len(entropies))

    # Top experts
    sorted_experts = sorted(range(n_experts), key=lambda i: usage_counts[i], reverse=True)
    top_experts = [
        {"expert_id": eid, "count": usage_counts[eid],
         "fraction": usage_counts[eid] / total_selections if total_selections > 0 else 0.0}
        for eid in sorted_experts[:20]
    ]

    # Bottom experts (least used)
    bottom_experts = [
        {"expert_id": eid, "count": usage_counts[eid],
         "fraction": usage_counts[eid] / total_selections if total_selections > 0 else 0.0}
        for eid in sorted_experts[-10:]
    ]

    return {
        "status": "ok",
        "n_experts": n_experts,
        "n_experts_per_tok": k,
        "n_moe_layers": n_moe_layers,
        "total_selections": total_selections,
        "usage_balance": round(usage_balance, 4),
        "usage_entropy": round(usage_entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "top_experts": top_experts,
        "bottom_experts": bottom_experts,
        "layer_routing_entropies": [round(e, 4) for e in layer_routing_entropies],
        "usage_distribution": [round(d, 6) for d in usage_dist],
    }


# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_results(results, info, output_dir, label):
    """Generate comprehensive visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available — skipping plots")
        return

    cats = list(PROMPT_CATEGORIES.keys())
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]

    # Determine grid layout based on whether we have MoE data
    has_moe = bool(results.get("overall", {}).get("expert_routing", {}).get("status") == "ok")
    n_rows = 6 if has_moe else 5

    fig = plt.figure(figsize=(24, 6 * n_rows))
    safe_label = label.replace("/", "-").replace(" ", "_")
    total_tokens = results.get("total_tokens", 0)
    overall = results.get("overall", {}).get("cp", {})

    fig.suptitle(
        f"Causal Primitives: {label}\n"
        f"{n_layers}L × {n_heads}H | {total_tokens} tokens"
        + (f" | MoE: {info['n_experts']} experts" if info["has_moe"] else ""),
        fontsize=14, fontweight="bold"
    )

    gs = fig.add_gridspec(n_rows, 2, hspace=0.4, wspace=0.3)

    # ── Panel 1: Overall CP heatmap ──
    ax = fig.add_subplot(gs[0, 0])
    if overall.get("status") == "ok":
        cp_map = np.array(overall["head_cp"])
        im = ax.imshow(cp_map, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"Overall Head CP — {overall.get('hierarchy', '?')}")
        plt.colorbar(im, ax=ax, label="CP")

    # ── Panel 2: Layer CP by category ──
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(n_layers)
    n_cats = len(cats)
    w = 0.8 / n_cats
    colors = plt.cm.tab10(np.linspace(0, 1, n_cats))
    for idx, cat in enumerate(cats):
        r = results.get("categories", {}).get(cat, {}).get("cp", {})
        if r.get("status") == "ok":
            ax.bar(x + idx * w - 0.4, r["layer_cp"], w,
                   label=cat[:8], color=colors[idx], alpha=0.7)
    ax.set_xlabel("Layer"); ax.set_ylabel("CP")
    ax.set_title("Layer CP by Category")
    ax.legend(fontsize=6, ncol=2)

    # ── Panels 3-4: Factual vs Hallucination ──
    for panel_idx, cat in enumerate(["factual", "hallucination"]):
        ax = fig.add_subplot(gs[1, panel_idx])
        r = results.get("categories", {}).get(cat, {}).get("cp", {})
        if r.get("status") == "ok":
            im = ax.imshow(np.array(r["head_cp"]), aspect="auto",
                           cmap="viridis", interpolation="nearest")
            plt.colorbar(im, ax=ax, label="CP")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"{cat.title()} Head CP")

    # ── Panels 5-6: Code vs Creative ──
    for panel_idx, cat in enumerate(["code", "creative"]):
        ax = fig.add_subplot(gs[2, panel_idx])
        r = results.get("categories", {}).get(cat, {}).get("cp", {})
        if r.get("status") == "ok":
            im = ax.imshow(np.array(r["head_cp"]), aspect="auto",
                           cmap="viridis", interpolation="nearest")
            plt.colorbar(im, ax=ax, label="CP")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title(f"{cat.title()} Head CP")

    # ── Panel 7: CP shift halluc - factual ──
    ax = fig.add_subplot(gs[3, 0])
    rf = results.get("categories", {}).get("factual", {}).get("cp", {})
    rh = results.get("categories", {}).get("hallucination", {}).get("cp", {})
    if rf.get("status") == "ok" and rh.get("status") == "ok":
        shift = np.array(rh["head_cp"]) - np.array(rf["head_cp"])
        vmax = max(abs(shift.min()), abs(shift.max())) or 0.01
        im = ax.imshow(shift, aspect="auto", cmap="RdBu_r",
                       interpolation="nearest", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="ΔCP")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("CP Shift: Hallucination − Factual")

    # ── Panel 8: CP shift code - creative ──
    ax = fig.add_subplot(gs[3, 1])
    rc = results.get("categories", {}).get("code", {}).get("cp", {})
    rcr = results.get("categories", {}).get("creative", {}).get("cp", {})
    if rc.get("status") == "ok" and rcr.get("status") == "ok":
        shift = np.array(rc["head_cp"]) - np.array(rcr["head_cp"])
        vmax = max(abs(shift.min()), abs(shift.max())) or 0.01
        im = ax.imshow(shift, aspect="auto", cmap="RdBu_r",
                       interpolation="nearest", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="ΔCP")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("CP Shift: Code − Creative")

    # ── Panel 9: Emergence by category ──
    ax = fig.add_subplot(gs[4, 0])
    cat_names = []
    emergences = []
    for cat in cats:
        r = results.get("categories", {}).get(cat, {}).get("cp", {})
        if r.get("status") == "ok":
            cat_names.append(cat[:8])
            emergences.append(r.get("emergence", 0.0))
    if cat_names:
        ax.bar(range(len(cat_names)), emergences, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(cat_names)))
        ax.set_xticklabels(cat_names, rotation=45, fontsize=8)
    ax.set_ylabel("Emergent Complexity")
    ax.set_title("Emergence by Category")

    # ── Panel 10: Summary ──
    ax = fig.add_subplot(gs[4, 1])
    ax.axis("off")
    lines = [
        f"Model: {label}",
        f"Architecture: {n_layers}L × {n_heads}H",
        f"Hidden dim: {info['hidden_dim']}",
        f"KV heads: {info['n_kv_heads']} (GQA ratio: {info['n_heads']//max(info['n_kv_heads'],1)}:1)",
        f"Tokens analyzed: {total_tokens}",
    ]
    if info["has_moe"]:
        lines.extend([
            f"MoE: {info['n_experts']} experts, top-{info['n_experts_per_tok']}",
        ])
    if overall.get("status") == "ok":
        lines.extend([
            "",
            f"Hierarchy: {overall.get('hierarchy', '?')}",
            f"Emergence: {overall.get('emergence', 0):.6f}",
            f"S_path:    {overall.get('S_path_norm', 0):.4f}",
            f"S_row_bar: {overall.get('S_row_bar', 0):.4f}",
            "",
            "Top 5 causal heads:",
        ])
        for h in overall.get("top_heads", [])[:5]:
            lines.append(f"  L{h['layer']}H{h['head']}: CP={h['cp']:.4f} "
                         f"det={h['determinism']:.4f} E={h['mean_energy']:.1f}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, fontfamily="monospace", verticalalignment="top")

    # ── MoE Panels (if applicable) ──
    if has_moe:
        expert_report = results["overall"]["expert_routing"]

        # Panel 11: Expert usage distribution
        ax = fig.add_subplot(gs[5, 0])
        usage = expert_report.get("usage_distribution", [])
        if usage:
            # Sort by usage for cleaner visualization
            sorted_usage = sorted(enumerate(usage), key=lambda x: x[1], reverse=True)
            expert_ids = [x[0] for x in sorted_usage[:40]]  # Top 40
            usages = [x[1] for x in sorted_usage[:40]]
            ax.bar(range(len(usages)), usages, color="coral", alpha=0.8)
            ax.set_xlabel("Expert (ranked by usage)")
            ax.set_ylabel("Selection Fraction")
            ax.set_title(f"Expert Usage Distribution (top 40 of {info['n_experts']})\n"
                         f"Balance: {expert_report.get('usage_balance', 0):.3f}")
            # Add expert IDs as labels for top 10
            if len(expert_ids) > 10:
                ax.set_xticks(range(0, len(expert_ids), max(1, len(expert_ids)//10)))
                ax.set_xticklabels(
                    [f"E{expert_ids[i]}" for i in range(0, len(expert_ids),
                                                         max(1, len(expert_ids)//10))],
                    fontsize=7, rotation=45
                )

        # Panel 12: Per-MoE-layer routing entropy
        ax = fig.add_subplot(gs[5, 1])
        layer_ent = expert_report.get("layer_routing_entropies", [])
        if layer_ent:
            ax.bar(range(len(layer_ent)), layer_ent, color="mediumpurple", alpha=0.8)
            ax.set_xlabel("MoE Layer Index")
            ax.set_ylabel("Routing Entropy (bits)")
            ax.set_title("Router Confidence by Layer\n(Lower = more decisive routing)")
            ax.axhline(y=expert_report.get("max_entropy", 0), color="red",
                       linestyle="--", alpha=0.5, label=f"Max entropy ({expert_report.get('max_entropy', 0):.1f})")
            ax.legend(fontsize=8)

    plot_path = os.path.join(output_dir, f"cp_mlx_{safe_label.lower()}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MLX Causal Primitives Analysis")
    parser.add_argument("--model", required=True,
                        help="Model path or HuggingFace ID (e.g., mlx-community/Qwen3.5-35B-A3B-4bit)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per prompt (default 1 — each token in prompt is an observation)")
    parser.add_argument("--bins", type=int, default=16,
                        help="Discretization bins for CP (default 16)")
    parser.add_argument("--label", default=None,
                        help="Human-readable model label (auto-detected if not set)")
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        print("ERROR: mlx and mlx_lm are required.")
        print("  pip install mlx mlx-lm")
        sys.exit(1)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Load model ──
    print("=" * 70)
    print(f"Loading model: {args.model}")
    print("=" * 70)

    t0 = time.time()
    model, tokenizer = load(args.model)
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    # ── Detect structure ──
    info, inner, layers = detect_model_structure(model)
    label = args.label or args.model.split("/")[-1]

    print(f"\nModel structure:")
    print(f"  Layers:    {info['n_layers']}")
    print(f"  Heads:     {info['n_heads']} query, {info['n_kv_heads']} KV")
    print(f"  Hidden:    {info['hidden_dim']}")
    print(f"  Head dim:  {info['head_dim']}")
    print(f"  MoE:       {'Yes' if info['has_moe'] else 'No'}")
    if info["has_moe"]:
        print(f"  Experts:   {info['n_experts']} total, top-{info['n_experts_per_tok']} per token")

    print(f"\nCategories: {list(PROMPT_CATEGORIES.keys())}")
    print(f"Prompts:    {sum(len(v) for v in PROMPT_CATEGORIES.values())} total")
    print(f"Runs:       {args.runs}")
    print(f"Bins:       {args.bins}")

    # ── Run analysis per category ──
    all_results = {
        "model": args.model,
        "label": label,
        "info": {k: v for k, v in info.items() if k not in ("layers_path",)},
        "total_tokens": 0,
        "categories": {},
    }

    all_prompts = []
    category_prompts = {}

    for cat, prompts in PROMPT_CATEGORIES.items():
        expanded = prompts * args.runs
        category_prompts[cat] = expanded
        all_prompts.extend(expanded)

    # Per-category analysis
    for cat, prompts in category_prompts.items():
        t0 = time.time()
        print(f"\n  Probing: {cat:15s} ({len(prompts)} prompts)...", end="", flush=True)

        result = run_cp_analysis(model, inner, layers, info, tokenizer,
                                 prompts, n_bins=args.bins)

        all_results["categories"][cat] = {
            "cp": result["cp"],
            "cross_head": {
                "n_interactions": result["cross_head"].get("n_interactions", 0),
                "top_interactions": result["cross_head"].get("top_interactions", [])[:15],
            },
            "expert_routing": result.get("expert_routing", {}),
            "tokens": result["total_tokens"],
        }
        all_results["total_tokens"] += result["total_tokens"]

        elapsed = time.time() - t0
        tokens = result["total_tokens"]
        cp_r = result["cp"]
        if cp_r.get("status") == "ok":
            th = cp_r["top_heads"][0]
            print(f" {tokens:5d} tok  {elapsed:5.1f}s  "
                  f"top=L{th['layer']}H{th['head']}(CP={th['cp']:.4f})  "
                  f"hierarchy={cp_r['hierarchy']}")
        else:
            print(f" {tokens:5d} tok  {elapsed:5.1f}s  (insufficient data)")

    # ── Overall analysis (all prompts combined) ──
    print(f"\n  Probing: {'OVERALL':15s} ({len(all_prompts)} prompts)...", end="", flush=True)
    t0 = time.time()
    overall = run_cp_analysis(model, inner, layers, info, tokenizer,
                              all_prompts, n_bins=args.bins)
    elapsed = time.time() - t0
    print(f" {overall['total_tokens']:5d} tok  {elapsed:5.1f}s")

    all_results["overall"] = {
        "cp": overall["cp"],
        "cross_head": {
            "n_interactions": overall["cross_head"].get("n_interactions", 0),
            "top_interactions": overall["cross_head"].get("top_interactions", [])[:30],
        },
        "expert_routing": overall.get("expert_routing", {}),
    }

    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {label}")
    print(f"{'=' * 70}")

    ov = overall["cp"]
    if ov.get("status") == "ok":
        print(f"  Hierarchy:  {ov['hierarchy']}")
        print(f"  Emergence:  {ov['emergence']:.6f}")
        print(f"  S_path:     {ov['S_path_norm']:.4f}")
        print(f"  S_row_bar:  {ov['S_row_bar']:.4f}")
        print(f"  Tokens:     {overall['total_tokens']}")

        print(f"\n  TOP 5 CAUSAL HEADS:")
        for h in ov["top_heads"][:5]:
            print(f"    L{h['layer']:2d}H{h['head']:2d}: CP={h['cp']:.4f}  "
                  f"det={h['determinism']:.4f}  spec={h['specificity']:.4f}  "
                  f"energy={h['mean_energy']:.1f}")

        print(f"\n  BOTTOM 5 HEADS:")
        for h in ov["bottom_heads"][:5]:
            print(f"    L{h['layer']:2d}H{h['head']:2d}: CP={h['cp']:.4f}  "
                  f"energy={h['mean_energy']:.1f}")

        print(f"\n  TOP 3 LAYERS:")
        layer_cp = ov["layer_cp"]
        sorted_layers = sorted(range(len(layer_cp)), key=lambda i: layer_cp[i], reverse=True)
        for i in sorted_layers[:3]:
            print(f"    Layer {i:2d}: CP={layer_cp[i]:.4f}")

        print(f"\n  PER-CATEGORY:")
        print(f"  {'Category':15s} {'Hierarchy':15s} {'Emerge':>8s} {'Top Head':>12s} {'TopCP':>8s}")
        print(f"  {'-' * 65}")
        for cat in PROMPT_CATEGORIES:
            cr = all_results["categories"].get(cat, {}).get("cp", {})
            if cr.get("status") == "ok":
                th = cr["top_heads"][0]
                print(f"  {cat:15s} {cr['hierarchy']:15s} {cr['emergence']:8.6f} "
                      f"  L{th['layer']}H{th['head']:2d}     {th['cp']:8.4f}")

        print(f"\n  TOP CROSS-HEAD LINKS:")
        cross = overall["cross_head"]
        for ix in cross.get("top_interactions", [])[:5]:
            print(f"    L{ix['cause_layer']:2d}H{ix['cause_head']:2d} -> "
                  f"L{ix['effect_layer']:2d}H{ix['effect_head']:2d}  "
                  f"MI={ix['mutual_info']:.4f}")

    # ── Expert routing summary ──
    er = overall.get("expert_routing", {})
    if er.get("status") == "ok":
        print(f"\n  MoE EXPERT ROUTING:")
        print(f"    Total selections: {er['total_selections']}")
        print(f"    Usage balance:    {er['usage_balance']:.4f} (1.0 = perfectly uniform)")
        print(f"    Usage entropy:    {er['usage_entropy']:.2f} / {er['max_entropy']:.2f} bits")
        print(f"\n    Most-used experts:")
        for e in er["top_experts"][:5]:
            print(f"      Expert {e['expert_id']:3d}: {e['fraction']:.4f} "
                  f"({e['count']} selections)")
        print(f"    Least-used experts:")
        for e in er["bottom_experts"][:3]:
            print(f"      Expert {e['expert_id']:3d}: {e['fraction']:.4f} "
                  f"({e['count']} selections)")

    # ── Save ──
    data_path = os.path.join(output_dir, f"cp_mlx_{label.lower().replace(' ', '_')}.json")
    with open(data_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nData saved: {data_path}")

    # ── Plot ──
    plot_results(all_results, info, output_dir, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
