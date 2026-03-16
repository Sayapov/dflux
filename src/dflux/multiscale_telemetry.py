#!/usr/bin/env python3
"""
Multi-Scale Telemetry — raw empirical signal capture at every scale.

No labels. No classifications. Pure telemetry.

Six signal types captured simultaneously across the full forward pass:

  1. RESIDUAL STREAM TRAJECTORY
     - Norm at each layer (how much energy is in the representation)
     - Direction change between consecutive layers (cosine distance)
     - Per-layer delta magnitude (how much each layer actually changes things)

  2. LOGIT LENS
     - Project residual stream onto unembedding matrix at each layer
     - Watch the model's "prediction" form in real time
     - Track top-k token probabilities and entropy at each layer

  3. CROSS-LAYER INFLUENCE
     - Cosine similarity between all layer pairs' residual contributions
     - Identifies which layers are doing similar vs orthogonal work
     - Captures long-range dependencies (layer 2 and layer 20 aligned?)

  4. MLP CONTRIBUTIONS
     - Per-layer MLP activation norms (separate from attention)
     - MLP vs attention energy ratio at each layer
     - Dead neuron count (fraction of near-zero activations in MLP hidden layer)
     - Activation outlier magnitude (max / mean ratio in each layer)

  5. ENTROPY CASCADE
     - Prediction entropy at each layer (via logit lens projection)
     - Tracks where the model narrows down its prediction
     - Entropy reduction rate between consecutive layers

  6. ACTIVATION OUTLIERS
     - Per-layer max activation magnitude
     - Outlier dimension index (which hidden dims carry disproportionate energy)
     - Gini coefficient of activation magnitudes (inequality measure)

All signals are raw numbers. Interpretation is your job.

Usage:
    from dflux.multiscale_telemetry import MultiScaleTelemetry

    telem = MultiScaleTelemetry.from_model(model, tokenizer)
    output = model.generate(input_ids, max_new_tokens=32)
    snapshot = telem.snapshot()

    # Raw data
    print(snapshot["residual_norms"])       # [n_layers] floats
    print(snapshot["logit_lens_entropy"])    # [n_layers] floats
    print(snapshot["cross_layer_sim"])       # [n_layers x n_layers] matrix

    # Full dump
    telem.save("telemetry_run.json")
    telem.detach()

Free instrument. MIT license.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

@dataclass
class TelemetryConfig:
    """Configuration for multi-scale telemetry capture.

    Args:
        logit_lens:         Enable logit lens projections (needs unembedding access).
        logit_lens_top_k:   How many top tokens to track per layer.
        cross_layer:        Enable cross-layer similarity matrix.
        mlp_internals:      Enable MLP activation analysis (hooks MLP hidden layer).
        entropy_cascade:    Enable entropy tracking (requires logit_lens).
        outlier_detection:  Enable activation outlier tracking.
        capture_residuals:  Store full residual vectors (memory heavy, off by default).
        max_snapshots:      Rolling buffer size for token snapshots.
    """
    logit_lens: bool = True
    logit_lens_top_k: int = 5
    cross_layer: bool = True
    mlp_internals: bool = True
    entropy_cascade: bool = True
    outlier_detection: bool = True
    capture_residuals: bool = False
    max_snapshots: int = 512


# ══════════════════════════════════════════════════════════════
# TOKEN SNAPSHOT — one per forward pass
# ══════════════════════════════════════════════════════════════

@dataclass
class TokenSnapshot:
    """Raw telemetry for a single token's forward pass through all layers."""
    token_idx: int
    timestamp: float  # time.monotonic()

    # ── Scale 1: Residual Stream ──
    residual_norms: List[float]           # [n_layers] L2 norm at each layer output
    residual_deltas: List[float]          # [n_layers] ||layer_out - layer_in||
    direction_changes: List[float]        # [n_layers-1] 1 - cos_sim(layer_i, layer_i+1)

    # ── Scale 2: Logit Lens ──
    logit_lens_top_tokens: Optional[List[List[Tuple[int, float]]]]  # [n_layers][top_k] (token_id, prob)
    logit_lens_entropy: Optional[List[float]]                       # [n_layers] entropy of projected logits
    logit_lens_top1_prob: Optional[List[float]]                     # [n_layers] probability of top prediction

    # ── Scale 3: Cross-Layer ──
    cross_layer_sim: Optional[List[List[float]]]  # [n_layers x n_layers] cosine similarity of deltas

    # ── Scale 4: MLP ──
    mlp_norms: Optional[List[float]]              # [n_layers] MLP output L2 norm
    attn_norms: Optional[List[float]]             # [n_layers] Attention output L2 norm
    mlp_attn_ratio: Optional[List[float]]         # [n_layers] MLP / (MLP + Attn) energy fraction
    mlp_dead_frac: Optional[List[float]]          # [n_layers] fraction of near-zero MLP activations
    mlp_outlier_ratio: Optional[List[float]]       # [n_layers] max / mean activation in MLP hidden

    # ── Scale 5: Entropy Cascade ──
    entropy_reduction: Optional[List[float]]       # [n_layers-1] entropy[i] - entropy[i+1]

    # ── Scale 6: Activation Outliers ──
    outlier_max_magnitude: Optional[List[float]]   # [n_layers] max |activation| in residual
    outlier_gini: Optional[List[float]]            # [n_layers] Gini coefficient of |activations|
    outlier_top_dims: Optional[List[List[int]]]    # [n_layers][top_5] indices of largest dims

    # ── Scale 7: Dimension Channels ──
    # Per-dimension energy across ALL layers — reveals persistent "channels" in residual stream
    dim_energy_per_layer: Optional[List[List[float]]]  # [n_layers][top_k_dims] energy of top dims
    dim_index_per_layer: Optional[List[List[int]]]     # [n_layers][top_k_dims] which dims they are
    persistent_dims: Optional[List[int]]               # dims that appear in top-k across >50% of layers
    persistent_dim_energies: Optional[List[float]]     # mean energy of each persistent dim across layers

    # ── Scale 8: Dilution Analysis (inspired by Moonshot AttnRes) ──
    # How much of each layer's contribution survives to the final output?
    # Standard residual connections dilute early layers as depth increases.
    dilution_survival: Optional[List[float]]           # [n_layers] cos_sim(delta_i, h_final) — alignment with output
    dilution_energy_frac: Optional[List[float]]        # [n_layers] ||proj(delta_i → h_final)|| / ||h_final||
    dilution_wasted_work: Optional[List[float]]        # [n_layers] ||delta_i|| * (1 - |survival|) — energy that doesn't reach output
    dilution_cumulative_drift: Optional[List[float]]   # [n_layers] cumulative direction change from layer 0

    # ── Layer metadata (for hybrid architectures) ──
    layer_types: Optional[List[str]] = None  # [n_layers] "full_attention" / "linear_attention" / "unknown"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "token_idx": self.token_idx,
            "timestamp": self.timestamp,
            "residual_norms": self.residual_norms,
            "residual_deltas": self.residual_deltas,
            "direction_changes": self.direction_changes,
        }
        for attr in [
            "logit_lens_top_tokens", "logit_lens_entropy", "logit_lens_top1_prob",
            "cross_layer_sim",
            "mlp_norms", "attn_norms", "mlp_attn_ratio", "mlp_dead_frac", "mlp_outlier_ratio",
            "entropy_reduction",
            "outlier_max_magnitude", "outlier_gini", "outlier_top_dims",
            "dim_energy_per_layer", "dim_index_per_layer",
            "persistent_dims", "persistent_dim_energies",
            "dilution_survival", "dilution_energy_frac",
            "dilution_wasted_work", "dilution_cumulative_drift",
            "layer_types",
        ]:
            val = getattr(self, attr)
            if val is not None:
                d[attr] = val
        return d


# ══════════════════════════════════════════════════════════════
# MULTI-SCALE TELEMETRY
# ══════════════════════════════════════════════════════════════

class MultiScaleTelemetry:
    """Raw multi-scale signal capture for transformer forward passes.

    Hooks into every transformer layer and optionally into MLP sub-modules.
    Captures six classes of signal simultaneously with zero interpretation.
    """

    def __init__(
        self,
        n_layers: int,
        cfg: TelemetryConfig,
        unembedding: Optional[torch.Tensor] = None,
        ln_f: Optional[nn.Module] = None,
    ) -> None:
        self.n_layers = n_layers
        self.cfg = cfg
        self._unembedding = unembedding  # [vocab, hidden] or None
        self._ln_f = ln_f                # final layer norm or None

        # ── Buffers filled by hooks ──
        self._layer_outputs: List[Optional[torch.Tensor]] = [None] * n_layers
        self._layer_inputs: List[Optional[torch.Tensor]] = [None] * n_layers
        self._attn_outputs: List[Optional[torch.Tensor]] = [None] * n_layers
        self._mlp_outputs: List[Optional[torch.Tensor]] = [None] * n_layers
        self._mlp_hidden_acts: List[Optional[torch.Tensor]] = [None] * n_layers

        # ── Hook handles ──
        self._hooks: List = []
        self._model = None
        self._tokenizer = None

        # ── State ──
        self._token_count: int = 0
        self._forward_count: int = 0
        self.snapshots: List[TokenSnapshot] = []

    # ══════════════════════════════════════════════════════════
    # FACTORY
    # ══════════════════════════════════════════════════════════

    @classmethod
    def from_model(
        cls,
        model,
        tokenizer=None,
        cfg: Optional[TelemetryConfig] = None,
        **kwargs,
    ) -> "MultiScaleTelemetry":
        """Auto-detect architecture and attach hooks everywhere."""
        if cfg is None:
            cfg = TelemetryConfig(**kwargs)

        layers = cls._find_transformer_layers(model)
        n_layers = len(layers)
        if n_layers == 0:
            raise ValueError(
                "Could not auto-detect transformer layers. "
                "Supported: GPT-2, LLaMA, Mistral, Qwen, Falcon, MPT, GPT-NeoX."
            )

        # Find unembedding matrix
        unembedding = None
        ln_f = None
        if cfg.logit_lens:
            unembedding, ln_f = cls._find_unembedding(model)
            if unembedding is None:
                print("[telemetry] WARNING: Could not find unembedding matrix. "
                      "Logit lens disabled.")
                cfg.logit_lens = False
                cfg.entropy_cascade = False

        telem = cls(n_layers=n_layers, cfg=cfg, unembedding=unembedding, ln_f=ln_f)
        telem._model = model
        telem._tokenizer = tokenizer

        # Detect layer types for hybrid architectures (e.g., Qwen3.5)
        telem._layer_types = [cls._get_layer_type(layer) for layer in layers]
        is_hybrid = any(lt != "unknown" for lt in telem._layer_types)

        telem._attach_hooks(layers)

        print(f"[telemetry] Attached to {n_layers} layers")
        if is_hybrid:
            n_full = sum(1 for lt in telem._layer_types if lt == "full_attention")
            n_linear = sum(1 for lt in telem._layer_types if lt == "linear_attention")
            print(f"[telemetry] Hybrid architecture: {n_full} full attention, {n_linear} linear attention")
        signals = []
        signals.append("residual_stream")
        if cfg.logit_lens:
            signals.append("logit_lens")
        if cfg.cross_layer:
            signals.append("cross_layer")
        if cfg.mlp_internals:
            signals.append("mlp_internals")
        if cfg.entropy_cascade:
            signals.append("entropy_cascade")
        if cfg.outlier_detection:
            signals.append("outlier_detection")
        print(f"[telemetry] Active signals: {', '.join(signals)}")

        return telem

    # ══════════════════════════════════════════════════════════
    # ARCHITECTURE DETECTION
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _find_transformer_layers(model) -> list:
        """Find repeated transformer blocks."""
        candidates = [
            "transformer.h",           # GPT-2, GPT-J, Falcon
            "model.layers",            # LLaMA, Mistral, Qwen, Phi
            "transformer.blocks",      # MPT
            "gpt_neox.layers",         # GPT-NeoX, Pythia
            "encoder.layer",           # BERT, RoBERTa
            "decoder.layers",          # T5 decoder
            "layers",                  # Generic
        ]
        for path in candidates:
            obj = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "__len__") and len(obj) > 1:
                    return list(obj)
            except AttributeError:
                continue
        # Fallback: ModuleList with >4 same-type children
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 4:
                types = [type(m).__name__ for m in module]
                if len(set(types)) == 1:
                    return list(module)
        return []

    @staticmethod
    def _find_unembedding(model) -> Tuple[Optional[torch.Tensor], Optional[nn.Module]]:
        """Find the unembedding (lm_head) weight and final layer norm."""
        unembed = None
        ln_f = None

        # ── Unembedding ──
        # Common paths: lm_head.weight, embed_out.weight
        for path in ["lm_head", "embed_out", "output", "head"]:
            obj = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if hasattr(obj, "weight"):
                    unembed = obj.weight.detach()  # [vocab, hidden]
                    break
            except AttributeError:
                continue

        # Tied weights: check if input embedding == output embedding
        if unembed is None:
            for path in [
                "transformer.wte", "model.embed_tokens",
                "gpt_neox.embed_in", "embeddings.word_embeddings",
            ]:
                obj = model
                try:
                    for attr in path.split("."):
                        obj = getattr(obj, attr)
                    if hasattr(obj, "weight"):
                        unembed = obj.weight.detach()
                        break
                except AttributeError:
                    continue

        # ── Final LayerNorm ──
        for path in [
            "transformer.ln_f",      # GPT-2
            "model.norm",            # LLaMA
            "gpt_neox.final_layer_norm",  # NeoX/Pythia
            "transformer.norm_f",    # MPT
        ]:
            obj = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                ln_f = obj
                break
            except AttributeError:
                continue

        return unembed, ln_f

    @staticmethod
    def _find_attn_module(layer) -> Optional[nn.Module]:
        """Find the attention sub-module within a transformer block.

        Handles standard softmax attention AND hybrid architectures like
        Qwen3.5 which interleave Gated DeltaNet (linear_attn) with
        full softmax attention (self_attn).
        """
        for attr in ["attn", "attention", "self_attn", "self_attention",
                      "linear_attn", "temporal_block"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

    @staticmethod
    def _get_layer_type(layer) -> str:
        """Detect layer type for hybrid architectures.

        Returns:
            "full_attention" — standard softmax attention
            "linear_attention" — Gated DeltaNet or similar linear attention
            "unknown" — standard transformer (non-hybrid)
        """
        if hasattr(layer, "layer_type"):
            return str(layer.layer_type)
        if hasattr(layer, "linear_attn"):
            return "linear_attention"
        if hasattr(layer, "temporal_block"):
            return "linear_attention"
        return "unknown"

    @staticmethod
    def _find_mlp_module(layer) -> Optional[nn.Module]:
        """Find the MLP sub-module within a transformer block."""
        for attr in ["mlp", "feed_forward", "ffn", "ff"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

    @staticmethod
    def _find_mlp_activation(mlp) -> Tuple[Optional[nn.Module], str]:
        """Find the activation function inside an MLP.

        We need to hook AFTER the activation function (GELU/SiLU/ReLU)
        to see which neurons are actually dead (zero post-activation).
        Hooking the linear projection pre-activation gives all nonzero values.

        Returns (module, strategy):
            strategy = "act_fn"   → hooked the activation function directly
            strategy = "down_proj" → hooked the down projection input
            strategy = "mlp_output" → fallback, hook the whole MLP output
        """
        # Strategy 1: Hook the activation function directly
        # NeoX/Pythia: mlp.act  (GELU)
        # GPT-2: mlp.act  (GELU)
        # LLaMA: mlp.act_fn  (SiLU)
        for attr in ["act", "act_fn", "activation_fn", "activation"]:
            if hasattr(mlp, attr):
                act = getattr(mlp, attr)
                if isinstance(act, nn.Module):
                    return act, "act_fn"

        # Strategy 2: Hook the down projection (its INPUT is post-activation)
        # NeoX: mlp.dense_4h_to_h
        # GPT-2: mlp.c_proj
        # LLaMA: mlp.down_proj
        for attr in ["dense_4h_to_h", "c_proj", "down_proj", "w2", "fc2"]:
            if hasattr(mlp, attr):
                return getattr(mlp, attr), "down_proj"

        return None, "none"

    # ══════════════════════════════════════════════════════════
    # HOOK MANAGEMENT
    # ══════════════════════════════════════════════════════════

    def _attach_hooks(self, layers: list) -> None:
        """Attach all hooks: layer-level + optional sub-module hooks."""
        self._hooks = []

        for i, layer in enumerate(layers):
            # ── Main layer hook (captures residual stream in/out) ──
            h = layer.register_forward_hook(self._make_layer_hook(i))
            self._hooks.append(h)

            # ── Attention sub-module hook ──
            attn = self._find_attn_module(layer)
            if attn is not None:
                h = attn.register_forward_hook(self._make_attn_hook(i))
                self._hooks.append(h)

            # ── MLP sub-module hook ──
            mlp = self._find_mlp_module(layer)
            if mlp is not None:
                h = mlp.register_forward_hook(self._make_mlp_hook(i))
                self._hooks.append(h)

                # ── MLP post-activation hook (for dead neuron / outlier analysis) ──
                if self.cfg.mlp_internals:
                    act_module, strategy = self._find_mlp_activation(mlp)
                    if act_module is not None:
                        if strategy == "act_fn":
                            # Hook activation fn output directly — this IS post-activation
                            h = act_module.register_forward_hook(
                                self._make_mlp_hidden_hook(i)
                            )
                        else:
                            # Hook down_proj — capture its INPUT (= post-activation hidden)
                            h = act_module.register_forward_hook(
                                self._make_mlp_down_input_hook(i)
                            )
                        self._hooks.append(h)

        # After all layers fire for one forward pass, process the snapshot
        # We detect completion by counting how many layer hooks have fired

    def _extract_hidden(self, output) -> Optional[torch.Tensor]:
        """Extract hidden states tensor from module output."""
        if isinstance(output, tuple):
            h = output[0]
        elif isinstance(output, dict):
            h = output.get("hidden_states", output.get("last_hidden_state"))
            if h is None:
                return None
        else:
            h = output
        if not isinstance(h, torch.Tensor):
            return None
        return h

    def _make_layer_hook(self, layer_idx: int):
        """Hook for the full transformer block — captures residual stream."""
        def hook(module, input, output):
            with torch.no_grad():
                # Output residual
                out = self._extract_hidden(output)
                if out is None:
                    return
                if out.dim() == 3:
                    out_vec = out[:, -1, :].detach()  # [batch, hidden]
                elif out.dim() == 2:
                    out_vec = out.detach()
                else:
                    return

                # Input residual
                inp = input[0] if isinstance(input, tuple) else input
                if isinstance(inp, torch.Tensor):
                    if inp.dim() == 3:
                        inp_vec = inp[:, -1, :].detach()
                    elif inp.dim() == 2:
                        inp_vec = inp.detach()
                    else:
                        inp_vec = None
                else:
                    inp_vec = None

                self._layer_outputs[layer_idx] = out_vec
                self._layer_inputs[layer_idx] = inp_vec

                # Check if all layers have fired
                if layer_idx == self.n_layers - 1:
                    self._on_forward_complete()

        return hook

    def _make_attn_hook(self, layer_idx: int):
        """Hook for attention sub-module output."""
        def hook(module, input, output):
            with torch.no_grad():
                out = self._extract_hidden(output)
                if out is None:
                    return
                if out.dim() == 3:
                    self._attn_outputs[layer_idx] = out[:, -1, :].detach()
                elif out.dim() == 2:
                    self._attn_outputs[layer_idx] = out.detach()
        return hook

    def _make_mlp_hook(self, layer_idx: int):
        """Hook for MLP sub-module output."""
        def hook(module, input, output):
            with torch.no_grad():
                out = self._extract_hidden(output)
                if out is None:
                    return
                if out.dim() == 3:
                    self._mlp_outputs[layer_idx] = out[:, -1, :].detach()
                elif out.dim() == 2:
                    self._mlp_outputs[layer_idx] = out.detach()
        return hook

    def _make_mlp_hidden_hook(self, layer_idx: int):
        """Hook for MLP post-activation values (output of activation function)."""
        def hook(module, input, output):
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    if output.dim() == 3:
                        self._mlp_hidden_acts[layer_idx] = output[:, -1, :].detach()
                    elif output.dim() == 2:
                        self._mlp_hidden_acts[layer_idx] = output.detach()
        return hook

    def _make_mlp_down_input_hook(self, layer_idx: int):
        """Hook for down projection — captures its INPUT which is post-activation."""
        def hook(module, input, output):
            with torch.no_grad():
                inp = input[0] if isinstance(input, tuple) else input
                if isinstance(inp, torch.Tensor):
                    if inp.dim() == 3:
                        self._mlp_hidden_acts[layer_idx] = inp[:, -1, :].detach()
                    elif inp.dim() == 2:
                        self._mlp_hidden_acts[layer_idx] = inp.detach()
        return hook

    # ══════════════════════════════════════════════════════════
    # FORWARD PASS PROCESSING
    # ══════════════════════════════════════════════════════════

    def _on_forward_complete(self) -> None:
        """Called when all layer hooks have fired for one forward pass.
        Computes all telemetry signals and creates a snapshot."""
        self._forward_count += 1

        # Only process if we have data for all layers
        if any(x is None for x in self._layer_outputs):
            return

        ts = time.monotonic()

        # ── Scale 1: Residual Stream Trajectory ──
        residual_norms = []
        residual_deltas = []
        direction_changes = []

        for i in range(self.n_layers):
            out = self._layer_outputs[i]
            inp = self._layer_inputs[i]

            # Norm
            norm = float(out.norm().cpu())
            residual_norms.append(norm)

            # Delta
            if inp is not None:
                delta = float((out - inp).norm().cpu())
            else:
                delta = 0.0
            residual_deltas.append(delta)

        # Direction changes (cosine distance between consecutive layer outputs)
        for i in range(1, self.n_layers):
            prev = self._layer_outputs[i - 1].float()
            curr = self._layer_outputs[i].float()
            cos = F.cosine_similarity(prev, curr, dim=-1)
            direction_changes.append(float((1.0 - cos).mean().cpu()))

        # ── Scale 2: Logit Lens ──
        logit_lens_top_tokens = None
        logit_lens_entropy = None
        logit_lens_top1_prob = None

        if self.cfg.logit_lens and self._unembedding is not None:
            logit_lens_top_tokens = []
            logit_lens_entropy = []
            logit_lens_top1_prob = []

            for i in range(self.n_layers):
                hidden = self._layer_outputs[i].float()  # [batch, hidden]

                # Apply final layer norm if available
                if self._ln_f is not None:
                    try:
                        hidden = self._ln_f(hidden)
                    except Exception:
                        pass  # Some norms need different shapes

                # Project onto vocabulary: [batch, hidden] @ [hidden, vocab] = [batch, vocab]
                logits = hidden @ self._unembedding.float().T
                probs = F.softmax(logits, dim=-1)

                # Top-k
                topk_probs, topk_ids = probs[0].topk(self.cfg.logit_lens_top_k)
                top_tokens = [
                    (int(tid), float(tp))
                    for tid, tp in zip(topk_ids.cpu(), topk_probs.cpu())
                ]
                logit_lens_top_tokens.append(top_tokens)

                # Entropy
                log_probs = torch.log(probs + 1e-10)
                ent = float(-(probs * log_probs).sum(dim=-1).mean().cpu())
                logit_lens_entropy.append(ent)

                # Top-1 probability
                logit_lens_top1_prob.append(float(topk_probs[0].cpu()))

        # ── Scale 3: Cross-Layer Influence ──
        cross_layer_sim = None
        if self.cfg.cross_layer:
            # Compute delta vectors (what each layer added)
            deltas = []
            for i in range(self.n_layers):
                inp = self._layer_inputs[i]
                out = self._layer_outputs[i]
                if inp is not None:
                    d = (out - inp).float()
                else:
                    d = out.float()
                deltas.append(d)

            # Pairwise cosine similarity of deltas
            cross_layer_sim = []
            for i in range(self.n_layers):
                row = []
                for j in range(self.n_layers):
                    cos = F.cosine_similarity(deltas[i], deltas[j], dim=-1)
                    row.append(float(cos.mean().cpu()))
                cross_layer_sim.append(row)

        # ── Scale 4: MLP Contributions ──
        mlp_norms = None
        attn_norms = None
        mlp_attn_ratio = None
        mlp_dead_frac = None
        mlp_outlier_ratio = None

        if self.cfg.mlp_internals:
            mlp_norms = []
            attn_norms = []
            mlp_attn_ratio = []
            mlp_dead_frac = []
            mlp_outlier_ratio = []

            for i in range(self.n_layers):
                mlp_out = self._mlp_outputs[i]
                attn_out = self._attn_outputs[i]

                m_norm = float(mlp_out.norm().cpu()) if mlp_out is not None else 0.0
                a_norm = float(attn_out.norm().cpu()) if attn_out is not None else 0.0
                mlp_norms.append(m_norm)
                attn_norms.append(a_norm)

                total = m_norm + a_norm
                mlp_attn_ratio.append(m_norm / total if total > 0 else 0.5)

                # Dead neurons and outliers from MLP hidden activations
                mlp_hidden = self._mlp_hidden_acts[i]
                if mlp_hidden is not None:
                    acts = mlp_hidden.float().abs()
                    dead = float((acts < 1e-6).float().mean().cpu())
                    mean_act = float(acts.mean().cpu())
                    max_act = float(acts.max().cpu())
                    mlp_dead_frac.append(dead)
                    mlp_outlier_ratio.append(
                        max_act / mean_act if mean_act > 0 else 0.0
                    )
                else:
                    mlp_dead_frac.append(None)
                    mlp_outlier_ratio.append(None)

        # ── Scale 5: Entropy Cascade ──
        entropy_reduction = None
        if self.cfg.entropy_cascade and logit_lens_entropy is not None:
            entropy_reduction = []
            for i in range(1, len(logit_lens_entropy)):
                entropy_reduction.append(
                    logit_lens_entropy[i - 1] - logit_lens_entropy[i]
                )

        # ── Scale 6: Activation Outliers ──
        outlier_max_magnitude = None
        outlier_gini = None
        outlier_top_dims = None

        if self.cfg.outlier_detection:
            outlier_max_magnitude = []
            outlier_gini = []
            outlier_top_dims = []

            for i in range(self.n_layers):
                out = self._layer_outputs[i].float()
                abs_out = out.abs().squeeze(0)  # [hidden]

                # Max magnitude
                outlier_max_magnitude.append(float(abs_out.max().cpu()))

                # Top-5 dimensions by magnitude
                topk_vals, topk_idx = abs_out.topk(min(5, abs_out.shape[0]))
                outlier_top_dims.append(topk_idx.cpu().tolist())

                # Gini coefficient
                sorted_vals, _ = abs_out.sort()
                n = sorted_vals.shape[0]
                if n > 0 and float(sorted_vals.sum()) > 0:
                    indices = torch.arange(1, n + 1, device=sorted_vals.device, dtype=sorted_vals.dtype)
                    gini = float(
                        (2.0 * (indices * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n).cpu()
                    )
                    outlier_gini.append(max(0.0, gini))
                else:
                    outlier_gini.append(0.0)

        # ── Scale 7: Dimension Channels ──
        # Track which hidden dimensions carry the most energy at each layer,
        # and find "persistent channels" — dims that dominate across many layers.
        dim_energy_per_layer = None
        dim_index_per_layer = None
        persistent_dims = None
        persistent_dim_energies = None

        if self.cfg.outlier_detection:
            top_k_dims = 20  # track top 20 dims per layer
            dim_energy_per_layer = []
            dim_index_per_layer = []

            # Count how often each dim appears in top-k across layers
            from collections import Counter
            dim_counts = Counter()
            dim_energies_accum = {}  # dim_idx → list of energies

            for i in range(self.n_layers):
                out = self._layer_outputs[i].float()
                abs_out = out.abs().squeeze(0)  # [hidden]

                topk_vals, topk_idx = abs_out.topk(min(top_k_dims, abs_out.shape[0]))
                energies = topk_vals.cpu().tolist()
                indices = topk_idx.cpu().tolist()

                dim_energy_per_layer.append(energies)
                dim_index_per_layer.append(indices)

                for idx, energy in zip(indices, energies):
                    dim_counts[idx] += 1
                    if idx not in dim_energies_accum:
                        dim_energies_accum[idx] = []
                    dim_energies_accum[idx].append(energy)

            # Persistent dims: appear in top-k in >50% of layers
            threshold = self.n_layers * 0.5
            persistent_dims = []
            persistent_dim_energies = []
            for dim_idx, count in dim_counts.most_common():
                if count >= threshold:
                    persistent_dims.append(dim_idx)
                    persistent_dim_energies.append(
                        sum(dim_energies_accum[dim_idx]) / len(dim_energies_accum[dim_idx])
                    )
                else:
                    break  # Counter is sorted by count, so we can stop

        # ── Scale 8: Dilution Analysis ──
        # For each layer: how much of its delta survives to the final output?
        # This directly measures the problem AttnRes (Moonshot 2026) addresses:
        # standard residual connections dilute early layer contributions.
        dilution_survival = None
        dilution_energy_frac = None
        dilution_wasted_work = None
        dilution_cumulative_drift = None

        h_final = self._layer_outputs[self.n_layers - 1].float()
        h_final_norm = h_final.norm()

        if h_final_norm > 0:
            dilution_survival = []
            dilution_energy_frac = []
            dilution_wasted_work = []
            dilution_cumulative_drift = []

            h_first = self._layer_outputs[0].float()

            for i in range(self.n_layers):
                inp = self._layer_inputs[i]
                out = self._layer_outputs[i]

                # Delta: what this layer wrote
                if inp is not None:
                    delta = (out - inp).float()
                else:
                    delta = out.float()

                delta_norm = delta.norm()

                # Survival: cosine similarity between this layer's delta and final output
                # +1 = perfectly aligned, 0 = orthogonal, -1 = opposing
                if delta_norm > 0:
                    survival = float(
                        F.cosine_similarity(delta, h_final, dim=-1).mean().cpu()
                    )
                else:
                    survival = 0.0
                dilution_survival.append(survival)

                # Energy fraction: how much of the final output this layer's delta explains
                # ||proj(delta → h_final)|| / ||h_final||
                # = (delta · h_final) / ||h_final||²  * ||h_final||
                # = (delta · h_final) / ||h_final||
                if delta_norm > 0 and h_final_norm > 0:
                    proj_magnitude = float(
                        (delta * h_final).sum() / (h_final_norm ** 2) * h_final_norm
                    )
                    energy_frac = abs(proj_magnitude) / float(h_final_norm)
                else:
                    energy_frac = 0.0
                dilution_energy_frac.append(float(energy_frac))

                # Wasted work: energy that doesn't contribute to final output
                # = ||delta|| * (1 - |cos_sim|)
                # High wasted work = the layer is doing a lot but it doesn't survive
                wasted = float(delta_norm) * (1.0 - abs(survival))
                dilution_wasted_work.append(wasted)

                # Cumulative drift: how far has the residual drifted from initial direction?
                out_f = self._layer_outputs[i].float()
                if h_first.norm() > 0 and out_f.norm() > 0:
                    drift = float(
                        (1.0 - F.cosine_similarity(h_first, out_f, dim=-1)).mean().cpu()
                    )
                else:
                    drift = 0.0
                dilution_cumulative_drift.append(drift)

        # ── Assemble Snapshot ──
        snapshot = TokenSnapshot(
            token_idx=self._token_count,
            timestamp=ts,
            residual_norms=residual_norms,
            residual_deltas=residual_deltas,
            direction_changes=direction_changes,
            logit_lens_top_tokens=logit_lens_top_tokens,
            logit_lens_entropy=logit_lens_entropy,
            logit_lens_top1_prob=logit_lens_top1_prob,
            cross_layer_sim=cross_layer_sim,
            mlp_norms=mlp_norms,
            attn_norms=attn_norms,
            mlp_attn_ratio=mlp_attn_ratio,
            mlp_dead_frac=mlp_dead_frac,
            mlp_outlier_ratio=mlp_outlier_ratio,
            entropy_reduction=entropy_reduction,
            outlier_max_magnitude=outlier_max_magnitude,
            outlier_gini=outlier_gini,
            outlier_top_dims=outlier_top_dims,
            dim_energy_per_layer=dim_energy_per_layer,
            dim_index_per_layer=dim_index_per_layer,
            persistent_dims=persistent_dims,
            persistent_dim_energies=persistent_dim_energies,
            dilution_survival=dilution_survival,
            dilution_energy_frac=dilution_energy_frac,
            dilution_wasted_work=dilution_wasted_work,
            dilution_cumulative_drift=dilution_cumulative_drift,
            layer_types=self._layer_types if any(lt != "unknown" for lt in self._layer_types) else None,
        )

        self.snapshots.append(snapshot)
        self._token_count += 1

        # Rolling buffer
        if len(self.snapshots) > self.cfg.max_snapshots:
            self.snapshots.pop(0)

        # Clear buffers
        self._clear_buffers()

    def _clear_buffers(self):
        """Reset per-token buffers."""
        for i in range(self.n_layers):
            self._layer_outputs[i] = None
            self._layer_inputs[i] = None
            self._attn_outputs[i] = None
            self._mlp_outputs[i] = None
            self._mlp_hidden_acts[i] = None

    # ══════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════

    def snapshot(self, idx: int = -1) -> Dict[str, Any]:
        """Get a single token snapshot as a dict."""
        if not self.snapshots:
            return {}
        return self.snapshots[idx].to_dict()

    def all_snapshots(self) -> List[Dict[str, Any]]:
        """Get all snapshots as a list of dicts."""
        return [s.to_dict() for s in self.snapshots]

    def aggregate(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all captured snapshots.

        Returns mean, std, min, max for each signal across tokens.
        This is the network-scale view.
        """
        if not self.snapshots:
            return {}

        n = len(self.snapshots)
        agg = {
            "n_tokens": n,
            "n_layers": self.n_layers,
        }

        # Helper: aggregate a list-of-lists signal
        def agg_signal(name: str):
            vals = []
            for s in self.snapshots:
                v = getattr(s, name)
                if v is not None:
                    vals.append(v)
            if not vals:
                return
            import numpy as np
            arr = np.array(vals, dtype=float)  # [n_tokens, n_layers]
            agg[f"{name}_mean"] = arr.mean(axis=0).tolist()
            agg[f"{name}_std"] = arr.std(axis=0).tolist()
            agg[f"{name}_min"] = arr.min(axis=0).tolist()
            agg[f"{name}_max"] = arr.max(axis=0).tolist()

        for signal in [
            "residual_norms", "residual_deltas", "direction_changes",
            "logit_lens_entropy", "logit_lens_top1_prob",
            "mlp_norms", "attn_norms", "mlp_attn_ratio",
            "outlier_max_magnitude", "outlier_gini",
            "entropy_reduction",
            "dilution_survival", "dilution_energy_frac",
            "dilution_wasted_work", "dilution_cumulative_drift",
        ]:
            agg_signal(signal)

        # Cross-layer similarity: average across tokens
        sims = []
        for s in self.snapshots:
            if s.cross_layer_sim is not None:
                sims.append(s.cross_layer_sim)
        if sims:
            import numpy as np
            agg["cross_layer_sim_mean"] = np.mean(sims, axis=0).tolist()

        # MLP dead fraction (may have Nones)
        dead_vals = []
        for s in self.snapshots:
            if s.mlp_dead_frac is not None:
                row = [v if v is not None else 0.0 for v in s.mlp_dead_frac]
                dead_vals.append(row)
        if dead_vals:
            import numpy as np
            arr = np.array(dead_vals, dtype=float)
            agg["mlp_dead_frac_mean"] = arr.mean(axis=0).tolist()

        # ── Dimension Channels ──
        # Find dims that persistently dominate across tokens AND layers
        from collections import Counter
        all_persistent = Counter()
        persistent_energies = {}
        for s in self.snapshots:
            if s.persistent_dims is not None:
                for dim_idx, energy in zip(s.persistent_dims, s.persistent_dim_energies):
                    all_persistent[dim_idx] += 1
                    if dim_idx not in persistent_energies:
                        persistent_energies[dim_idx] = []
                    persistent_energies[dim_idx].append(energy)

        if all_persistent:
            # Dims that are persistent across >50% of tokens
            token_threshold = n * 0.5
            stable_channels = []
            for dim_idx, count in all_persistent.most_common():
                if count >= token_threshold:
                    stable_channels.append({
                        "dim": dim_idx,
                        "token_frequency": count / n,
                        "mean_energy": sum(persistent_energies[dim_idx]) / len(persistent_energies[dim_idx]),
                        "std_energy": float(torch.tensor(persistent_energies[dim_idx]).std()),
                    })
            agg["stable_dimension_channels"] = stable_channels
            agg["n_stable_channels"] = len(stable_channels)

            # Build layer presence matrix for stable channels
            # For each stable dim, which layers does it appear in?
            if stable_channels:
                channel_layer_presence = {}
                for s in self.snapshots:
                    if s.dim_index_per_layer is not None:
                        for layer_idx, layer_dims in enumerate(s.dim_index_per_layer):
                            for dim_idx in layer_dims:
                                if any(ch["dim"] == dim_idx for ch in stable_channels):
                                    key = dim_idx
                                    if key not in channel_layer_presence:
                                        channel_layer_presence[key] = Counter()
                                    channel_layer_presence[key][layer_idx] += 1
                # Normalize by n_tokens
                agg["channel_layer_heatmap"] = {
                    str(dim_idx): {
                        str(layer): count / n
                        for layer, count in layer_counts.items()
                    }
                    for dim_idx, layer_counts in channel_layer_presence.items()
                }

        return agg

    def save(self, path: str) -> None:
        """Save all telemetry to JSON."""
        data = {
            "model": getattr(self._model, "name_or_path",
                     getattr(getattr(self._model, "config", None), "_name_or_path", "unknown")),
            "n_layers": self.n_layers,
            "n_tokens": len(self.snapshots),
            "layer_types": self._layer_types if any(lt != "unknown" for lt in self._layer_types) else None,
            "config": {
                "logit_lens": self.cfg.logit_lens,
                "cross_layer": self.cfg.cross_layer,
                "mlp_internals": self.cfg.mlp_internals,
                "entropy_cascade": self.cfg.entropy_cascade,
                "outlier_detection": self.cfg.outlier_detection,
            },
            "aggregate": self.aggregate(),
            "snapshots": self.all_snapshots(),
        }

        # Decode logit lens tokens if tokenizer available
        if self._tokenizer is not None:
            for snap in data["snapshots"]:
                if "logit_lens_top_tokens" in snap:
                    decoded = []
                    for layer_tokens in snap["logit_lens_top_tokens"]:
                        decoded.append([
                            (self._tokenizer.decode([tid]), prob)
                            for tid, prob in layer_tokens
                        ])
                    snap["logit_lens_top_tokens_decoded"] = decoded

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[telemetry] Saved {len(self.snapshots)} snapshots → {path}")

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._clear_buffers()
        print("[telemetry] Detached all hooks.")

    def reset(self) -> None:
        """Clear all captured data but keep hooks attached."""
        self.snapshots.clear()
        self._token_count = 0
        self._forward_count = 0
        self._clear_buffers()

    def summary(self) -> str:
        """Human-readable summary of captured telemetry."""
        if not self.snapshots:
            return "No data captured yet."

        lines = [
            f"Multi-Scale Telemetry: {len(self.snapshots)} tokens, {self.n_layers} layers",
            "",
        ]

        agg = self.aggregate()

        # Residual stream
        if "residual_norms_mean" in agg:
            norms = agg["residual_norms_mean"]
            lines.append(f"Residual norms:  min={min(norms):.1f}  max={max(norms):.1f}  "
                         f"growth={norms[-1]/norms[0]:.2f}x")

        if "residual_deltas_mean" in agg:
            deltas = agg["residual_deltas_mean"]
            peak_layer = deltas.index(max(deltas))
            lines.append(f"Layer deltas:    peak at L{peak_layer} ({max(deltas):.1f}), "
                         f"min at L{deltas.index(min(deltas))} ({min(deltas):.1f})")

        if "direction_changes_mean" in agg:
            dc = agg["direction_changes_mean"]
            peak = dc.index(max(dc))
            lines.append(f"Direction shifts: biggest pivot at L{peak+1} ({max(dc):.4f})")

        # Logit lens
        if "logit_lens_entropy_mean" in agg:
            ent = agg["logit_lens_entropy_mean"]
            lines.append(f"Entropy cascade: L0={ent[0]:.2f} → L{self.n_layers-1}={ent[-1]:.2f} "
                         f"(reduction: {ent[0]-ent[-1]:.2f})")

        if "logit_lens_top1_prob_mean" in agg:
            p = agg["logit_lens_top1_prob_mean"]
            lines.append(f"Top-1 confidence: L0={p[0]:.4f} → L{self.n_layers-1}={p[-1]:.4f}")

        # MLP vs Attention
        if "mlp_attn_ratio_mean" in agg:
            ratios = agg["mlp_attn_ratio_mean"]
            mlp_dom = sum(1 for r in ratios if r > 0.5)
            lines.append(f"MLP dominates in {mlp_dom}/{self.n_layers} layers")

        # Outliers
        if "outlier_gini_mean" in agg:
            gini = agg["outlier_gini_mean"]
            lines.append(f"Activation Gini: min={min(gini):.3f}  max={max(gini):.3f} "
                         f"(1.0 = all energy in one dim)")

        # Dilution analysis
        if "dilution_survival_mean" in agg:
            surv = agg["dilution_survival_mean"]
            wasted = agg["dilution_wasted_work_mean"]
            drift = agg["dilution_cumulative_drift_mean"]

            # Most diluted layers: high delta but low survival
            most_wasted = sorted(enumerate(wasted), key=lambda x: x[1], reverse=True)
            lines.append(f"\nDilution analysis:")
            lines.append(f"  Survival range: {min(surv):.3f} → {max(surv):.3f} "
                         f"(1.0 = fully aligned with output)")
            lines.append(f"  Cumulative drift: L0→L{self.n_layers-1} = {drift[-1]:.4f}")
            lines.append(f"  Most wasted work:")
            for idx, w in most_wasted[:5]:
                lines.append(f"    L{idx}: wasted={w:.1f}, survival={surv[idx]:+.3f}")

        # Dimension channels
        if "stable_dimension_channels" in agg:
            channels = agg["stable_dimension_channels"]
            lines.append(f"\nDimension channels: {len(channels)} stable channels "
                         f"(persistent across >50% of tokens AND layers)")
            for ch in channels[:10]:  # Show top 10
                lines.append(f"  dim {ch['dim']:>4d}: freq={ch['token_frequency']:.0%} "
                             f"energy={ch['mean_energy']:.2f} ±{ch['std_energy']:.2f}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# CONVENIENCE
# ══════════════════════════════════════════════════════════════

def quick_telemetry(
    model,
    tokenizer,
    prompt: str = "The quick brown fox jumps over the lazy dog.",
    max_new_tokens: int = 32,
    **kwargs,
) -> Dict[str, Any]:
    """One-liner: attach → generate → return aggregate telemetry."""
    telem = MultiScaleTelemetry.from_model(model, tokenizer, **kwargs)

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

    result = {
        "prompt": prompt,
        "summary": telem.summary(),
        "aggregate": telem.aggregate(),
        "snapshots": telem.all_snapshots(),
    }

    telem.detach()
    return result
