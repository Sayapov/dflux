#!/usr/bin/env python3
"""
Fine-Grained Inference Probe — per-head, per-module activation monitor.

Extends the base inference probe to capture:
  - Per-attention-head energy (which heads are working hardest)
  - Attention vs MLP decomposition (routing vs transformation stress)
  - Cross-head agreement/disagreement (entropy of head energy distribution)
  - Per-head residual deltas (how much each head changes the signal)

This gives you a detailed X-ray of what the model is doing at each token:
  - Which cognitive functions are engaged (different heads = different roles)
  - Whether the model is struggling to route (attn turbulent) or transform (MLP turbulent)
  - Whether heads agree (low entropy) or disagree (high entropy) about what's important

Usage:
    from dflux.fine_probe import FineProbe, FineProbeConfig

    probe = FineProbe.from_model(model)
    output = model.generate(input_ids, max_new_tokens=64)
    report = probe.report()

    # Per-token, per-head resolution
    for d in probe.diagnostics:
        print(f"token {d.token_idx}: attn_energy={d.attn_energy:.2f}, "
              f"mlp_energy={d.mlp_energy:.2f}, head_entropy={d.head_entropy:.3f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dflux.axe_ns import Regime, classify_regime, JState


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

@dataclass
class FineProbeConfig:
    """Configuration for fine-grained inference probe.

    Args:
        L_cut: Layer cutoff for head/tail split. 0 = auto (60%).
        window_tokens: Tokens per measurement window.
        j_func: Complexity functional ("tail_ratio", "energy").
        capture_attention_weights: If True, capture full attention weight
            matrices (expensive but shows what each head attends to).
    """
    L_cut: int = 0
    window_tokens: int = 16
    j_func: str = "tail_ratio"
    capture_attention_weights: bool = False


# ══════════════════════════════════════════════════════════════
# FINE-GRAINED TOKEN DIAGNOSTIC
# ══════════════════════════════════════════════════════════════

@dataclass
class FineTokenDiagnostic:
    """Per-token diagnostic with per-head, per-module resolution."""
    token_idx: int

    # Overall regime & risk (same as base probe)
    regime: str
    J: float
    hallucination_risk: float

    # Per-layer decomposition
    layer_attn_energy: List[float]    # Attention output norm² per layer
    layer_mlp_energy: List[float]     # MLP output norm² per layer
    layer_total_energy: List[float]   # Total layer output norm²

    # Per-head decomposition (flattened: [layer0_head0, layer0_head1, ..., layerN_headM])
    head_energies: List[List[float]]  # [n_layers][n_heads] — norm² per head per layer

    # Derived metrics
    attn_energy: float                # Total attention energy across all layers
    mlp_energy: float                 # Total MLP energy across all layers
    attn_mlp_ratio: float             # attn / (attn + mlp) — where is the work?
    head_entropy: float               # Entropy of head energy distribution (agreement)
    head_gini: float                  # Gini coefficient of head energies (concentration)

    # Flux
    flux: float
    delta_flux: float

    # Per-layer attention-focus metrics
    layer_head_entropy: List[float]   # Head entropy per layer

    # Energy decomposition
    E_tail: float
    E_total: float


# ══════════════════════════════════════════════════════════════
# FINE-GRAINED PROBE
# ══════════════════════════════════════════════════════════════

class FineProbe:
    """Fine-grained activation probe with per-head, per-module resolution.

    Hooks into attention outputs (per-head), MLP outputs, and full layer
    outputs to decompose the activation dynamics at maximum resolution.

    Architecture detection:
        GPT-2:   layer.attn (c_proj output), layer.mlp
        LLaMA:   layer.self_attn (o_proj output), layer.mlp
        Mistral: layer.self_attn (o_proj output), layer.mlp
        Qwen:    layer.self_attn, layer.mlp
    """

    def __init__(self, n_layers: int, n_heads: int, cfg: FineProbeConfig) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cfg = cfg
        if cfg.L_cut == 0:
            cfg.L_cut = max(0, int(0.6 * n_layers) - 1)

        # J state for regime tracking
        self.J = JState(window=cfg.window_tokens * 2)
        self._last_E_tail: Optional[float] = None
        self._window_accum: float = 0.0
        self._window_start: int = 0
        self._forward_count: int = 0

        # Per-layer buffers
        self._layer_attn_norms: List[Optional[float]] = [None] * n_layers
        self._layer_mlp_norms: List[Optional[float]] = [None] * n_layers
        self._layer_output_norms: List[Optional[float]] = [None] * n_layers
        self._layer_head_norms: List[Optional[List[float]]] = [None] * n_layers

        # Tracking which modules have fired
        self._attn_seen: int = 0
        self._mlp_seen: int = 0
        self._layer_seen: int = 0

        # Hooks
        self._hooks: list = []
        self._model = None

        # Output
        self.diagnostics: List[FineTokenDiagnostic] = []
        self.regime_history: List[Regime] = []

    # ── Factory ──────────────────────────────────────────────

    @classmethod
    def from_model(
        cls,
        model,
        cfg: Optional[FineProbeConfig] = None,
        **kwargs,
    ) -> "FineProbe":
        """Auto-detect architecture and attach fine-grained hooks."""
        if cfg is None:
            cfg = FineProbeConfig(**kwargs)

        layers = cls._find_transformer_layers(model)
        n_layers = len(layers)
        if n_layers == 0:
            raise ValueError("Could not auto-detect transformer layers.")

        # Detect number of attention heads
        n_heads = cls._find_n_heads(model, layers[0])

        probe = cls(n_layers=n_layers, n_heads=n_heads, cfg=cfg)
        probe._model = model
        probe._attach_fine_hooks(layers)
        return probe

    @staticmethod
    def _find_transformer_layers(model) -> list:
        """Find repeated transformer blocks."""
        candidates = [
            "transformer.h",        # GPT-2, GPT-J, Falcon
            "model.layers",         # LLaMA, Mistral, Qwen, Phi
            "transformer.blocks",   # MPT
            "encoder.layer",        # BERT
            "gpt_neox.layers",      # GPT-NeoX
            "layers",               # Generic
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

        import torch.nn as nn
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 4:
                types = [type(m).__name__ for m in module]
                if len(set(types)) == 1:
                    return list(module)
        return []

    @staticmethod
    def _find_n_heads(model, sample_layer) -> int:
        """Detect number of attention heads from model config or layer structure."""
        # Try model config first
        if hasattr(model, 'config'):
            for attr in ['num_attention_heads', 'n_head', 'num_heads']:
                if hasattr(model.config, attr):
                    return getattr(model.config, attr)

        # Try layer's attention module
        for attr in ['attn', 'self_attn', 'attention']:
            if hasattr(sample_layer, attr):
                attn_mod = getattr(sample_layer, attr)
                for a in ['num_heads', 'n_head', 'num_attention_heads']:
                    if hasattr(attn_mod, a):
                        return getattr(attn_mod, a)

        return 12  # Default fallback

    # ── Fine-grained hooks ────────────────────────────────────

    def _attach_fine_hooks(self, layers: list) -> None:
        """Attach hooks to attention submodules, MLP submodules, and full layers."""
        import torch

        self._hooks = []

        for i, layer in enumerate(layers):
            # ── Find attention submodule ──
            attn_mod = None
            for attr in ['attn', 'self_attn', 'attention']:
                if hasattr(layer, attr):
                    attn_mod = getattr(layer, attr)
                    break

            # ── Find MLP submodule ──
            mlp_mod = None
            for attr in ['mlp', 'feed_forward', 'ffn']:
                if hasattr(layer, attr):
                    mlp_mod = getattr(layer, attr)
                    break

            # Hook attention
            if attn_mod is not None:
                def make_attn_hook(layer_idx, attn_module):
                    def hook(module, input, output):
                        with torch.no_grad():
                            # Output can be tuple: (hidden_states, attn_weights, ...)
                            # or (hidden_states, present_kv) or just hidden_states
                            if isinstance(output, tuple):
                                hidden = output[0]
                            else:
                                hidden = output

                            if hidden.dim() == 3:
                                h = hidden[:, -1, :]  # Last token
                            elif hidden.dim() == 2:
                                h = hidden
                            else:
                                self._attn_seen += 1
                                return

                            norm = float(h.norm().cpu())
                            self._layer_attn_norms[layer_idx] = norm

                            # Per-head decomposition
                            # Reshape hidden [batch, hidden_dim] -> [batch, n_heads, head_dim]
                            head_dim = h.shape[-1] // self.n_heads
                            if h.shape[-1] % self.n_heads == 0 and head_dim > 0:
                                h_heads = h.view(-1, self.n_heads, head_dim)
                                head_norms = [
                                    float(h_heads[0, hi, :].norm().cpu())
                                    for hi in range(self.n_heads)
                                ]
                                self._layer_head_norms[layer_idx] = head_norms
                            else:
                                # Fallback: can't cleanly decompose
                                self._layer_head_norms[layer_idx] = [norm / self.n_heads] * self.n_heads

                        self._attn_seen += 1
                    return hook

                h = attn_mod.register_forward_hook(make_attn_hook(i, attn_mod))
                self._hooks.append(h)

            # Hook MLP
            if mlp_mod is not None:
                def make_mlp_hook(layer_idx):
                    def hook(module, input, output):
                        with torch.no_grad():
                            if isinstance(output, tuple):
                                hidden = output[0]
                            else:
                                hidden = output

                            if hidden.dim() == 3:
                                h = hidden[:, -1, :]
                            elif hidden.dim() == 2:
                                h = hidden
                            else:
                                self._mlp_seen += 1
                                return

                            norm = float(h.norm().cpu())
                            self._layer_mlp_norms[layer_idx] = norm

                        self._mlp_seen += 1
                    return hook

                h = mlp_mod.register_forward_hook(make_mlp_hook(i))
                self._hooks.append(h)

            # Hook full layer output
            def make_layer_hook(layer_idx):
                def hook(module, input, output):
                    with torch.no_grad():
                        if isinstance(output, tuple):
                            hidden = output[0]
                        else:
                            hidden = output

                        if hidden.dim() == 3:
                            h = hidden[:, -1, :]
                        elif hidden.dim() == 2:
                            h = hidden
                        else:
                            self._layer_seen += 1
                            return

                        norm = float(h.norm().cpu())
                        self._layer_output_norms[layer_idx] = norm

                    self._layer_seen += 1

                    # Trigger processing when all layers complete
                    if self._layer_seen >= self.n_layers:
                        self._auto_process()
                        self._layer_seen = 0
                        self._attn_seen = 0
                        self._mlp_seen = 0

                return hook

            h = layer.register_forward_hook(make_layer_hook(i))
            self._hooks.append(h)

    # ── Processing ────────────────────────────────────────────

    def _auto_process(self) -> None:
        """Called when all layers have completed one forward pass."""
        token_idx = self._forward_count
        self._forward_count += 1

        # Collect per-layer energies
        layer_attn_energy = []
        layer_mlp_energy = []
        layer_total_energy = []
        head_energies = []
        layer_head_entropy = []

        for i in range(self.n_layers):
            attn_n = self._layer_attn_norms[i] or 0.0
            mlp_n = self._layer_mlp_norms[i] or 0.0
            total_n = self._layer_output_norms[i] or 0.0

            layer_attn_energy.append(attn_n ** 2)
            layer_mlp_energy.append(mlp_n ** 2)
            layer_total_energy.append(total_n ** 2)

            # Per-head energies for this layer
            if self._layer_head_norms[i] is not None:
                h_norms = self._layer_head_norms[i]
                h_energies = [n ** 2 for n in h_norms]
            else:
                h_energies = [0.0] * self.n_heads

            head_energies.append(h_energies)

            # Head entropy for this layer (how spread out is the work?)
            h_total = sum(h_energies) + 1e-12
            h_probs = [e / h_total for e in h_energies]
            entropy = -sum(p * math.log(p + 1e-12) for p in h_probs)
            max_entropy = math.log(self.n_heads + 1e-12)
            layer_head_entropy.append(entropy / max_entropy if max_entropy > 0 else 0.0)

        # Aggregate metrics
        total_attn = sum(layer_attn_energy)
        total_mlp = sum(layer_mlp_energy)
        E_total = sum(layer_total_energy)
        E_tail = sum(layer_total_energy[self.cfg.L_cut + 1:])

        attn_mlp_ratio = total_attn / (total_attn + total_mlp + 1e-12)

        # Global head entropy (across ALL heads in ALL layers)
        all_head_energies = [e for layer_heads in head_energies for e in layer_heads]
        total_head_e = sum(all_head_energies) + 1e-12
        head_probs = [e / total_head_e for e in all_head_energies]
        global_entropy = -sum(p * math.log(p + 1e-12) for p in head_probs)
        max_global_entropy = math.log(len(all_head_energies) + 1e-12)
        head_entropy = global_entropy / max_global_entropy if max_global_entropy > 0 else 0.0

        # Gini coefficient of head energies (how concentrated is the work?)
        sorted_energies = sorted(all_head_energies)
        n = len(sorted_energies)
        if n > 0 and total_head_e > 1e-12:
            cumulative = sum((2 * (i + 1) - n - 1) * e for i, e in enumerate(sorted_energies))
            gini = cumulative / (n * total_head_e)
        else:
            gini = 0.0

        # Flux computation
        if self._last_E_tail is None:
            flux = 0.0
        else:
            dE = E_tail - self._last_E_tail
            flux = max(dE, 0.0)
        self._last_E_tail = E_tail

        self._window_accum += flux
        delta_flux = self._window_accum
        if (token_idx - self._window_start + 1) >= self.cfg.window_tokens:
            self._window_start = token_idx + 1
            self._window_accum = 0.0

        # Regime
        flux_ratio = flux / max(E_total, 1e-12)
        regime = classify_regime(E_tail, E_total, flux_ratio)
        self.regime_history.append(regime)

        # J
        if self.cfg.j_func == "tail_ratio":
            j_val = E_tail / max(E_total, 1e-12)
        else:
            j_val = E_total
        self.J.record(j_val)

        # Hallucination risk (enhanced with head/module info)
        risk = self._compute_risk(regime, E_tail, E_total, delta_flux,
                                   head_entropy, attn_mlp_ratio, gini)

        diagnostic = FineTokenDiagnostic(
            token_idx=token_idx,
            regime=regime.value,
            J=j_val,
            hallucination_risk=risk,
            layer_attn_energy=layer_attn_energy,
            layer_mlp_energy=layer_mlp_energy,
            layer_total_energy=layer_total_energy,
            head_energies=head_energies,
            attn_energy=total_attn,
            mlp_energy=total_mlp,
            attn_mlp_ratio=attn_mlp_ratio,
            head_entropy=head_entropy,
            head_gini=gini,
            flux=flux,
            delta_flux=delta_flux,
            layer_head_entropy=layer_head_entropy,
            E_tail=E_tail,
            E_total=E_total,
        )
        self.diagnostics.append(diagnostic)

        # Reset buffers
        self._layer_attn_norms = [None] * self.n_layers
        self._layer_mlp_norms = [None] * self.n_layers
        self._layer_output_norms = [None] * self.n_layers
        self._layer_head_norms = [None] * self.n_layers

    def _compute_risk(
        self,
        regime: Regime,
        E_tail: float,
        E_total: float,
        delta_flux: float,
        head_entropy: float,
        attn_mlp_ratio: float,
        gini: float,
    ) -> float:
        """Enhanced hallucination risk with fine-grained signals.

        Components:
          1. Regime severity (0-0.25)
          2. Tail fraction (0-0.20)
          3. J trend (0-0.15)
          4. Head entropy (0-0.15) — low entropy = few heads dominating = confident
          5. Head concentration/gini (0-0.10) — high gini = very few heads doing all work
          6. Attn/MLP imbalance (0-0.15) — extreme ratios are suspicious
        """
        # 1. Regime
        regime_score = {
            Regime.LAMINAR: 0.0,
            Regime.TRANSITIONAL: 0.06,
            Regime.TURBULENT: 0.17,
            Regime.CRITICAL: 0.25,
        }.get(regime, 0.0)

        # 2. Tail fraction
        tail_frac = E_tail / max(E_total, 1e-12)
        tail_score = min(tail_frac, 1.0) * 0.20

        # 3. J trend
        trend = self.J.trend
        trend_score = min(max(trend * 50, 0.0), 0.15)

        # 4. Head entropy — low entropy means a few heads are doing everything
        #    That's actually a sign of confident (possibly over-confident) processing
        #    High entropy = distributed = uncertain exploration
        entropy_score = head_entropy * 0.15

        # 5. Gini — high concentration = specialized processing = confident
        #    Low concentration = distributed = uncertain
        gini_score = (1.0 - gini) * 0.10  # Invert: low gini = high score

        # 6. Attn/MLP ratio — extreme imbalance
        #    Normal is ~0.3-0.7. Very high attn = routing-dominated (retrieval)
        #    Very high MLP = transformation-dominated (generation/fabrication)
        imbalance = abs(attn_mlp_ratio - 0.5) * 2.0  # 0 at 0.5, 1 at extremes
        imbalance_score = imbalance * 0.15

        return min(regime_score + tail_score + trend_score +
                   entropy_score + gini_score + imbalance_score, 1.0)

    # ── Report ───────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Generate comprehensive fine-grained report."""
        if not self.diagnostics:
            return {"status": "no_data"}

        risks = [d.hallucination_risk for d in self.diagnostics]
        n_tok = len(self.diagnostics)

        # Average per-layer attention vs MLP energy
        avg_attn_per_layer = [0.0] * self.n_layers
        avg_mlp_per_layer = [0.0] * self.n_layers
        avg_head_entropy_per_layer = [0.0] * self.n_layers

        # Average per-head energy [n_layers][n_heads]
        avg_head_energy = [[0.0] * self.n_heads for _ in range(self.n_layers)]

        for d in self.diagnostics:
            for i in range(self.n_layers):
                avg_attn_per_layer[i] += d.layer_attn_energy[i] / n_tok
                avg_mlp_per_layer[i] += d.layer_mlp_energy[i] / n_tok
                avg_head_entropy_per_layer[i] += d.layer_head_entropy[i] / n_tok
                for j in range(self.n_heads):
                    if i < len(d.head_energies) and j < len(d.head_energies[i]):
                        avg_head_energy[i][j] += d.head_energies[i][j] / n_tok

        # Regime distribution
        regime_counts = {}
        for d in self.diagnostics:
            regime_counts[d.regime] = regime_counts.get(d.regime, 0) + 1

        # Attn/MLP ratio trajectory
        attn_mlp_trajectory = [d.attn_mlp_ratio for d in self.diagnostics]
        head_entropy_trajectory = [d.head_entropy for d in self.diagnostics]
        gini_trajectory = [d.head_gini for d in self.diagnostics]

        # Find most active heads (highest average energy)
        head_rankings = []
        for i in range(self.n_layers):
            for j in range(self.n_heads):
                head_rankings.append({
                    "layer": i,
                    "head": j,
                    "avg_energy": avg_head_energy[i][j],
                })
        head_rankings.sort(key=lambda x: x["avg_energy"], reverse=True)

        return {
            "total_tokens": n_tok,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "mean_risk": sum(risks) / n_tok,
            "max_risk": max(risks),
            "regime_distribution": regime_counts,
            "J_final": self.J.current,
            "J_trend": self.J.trend,

            # Module decomposition
            "mean_attn_mlp_ratio": sum(attn_mlp_trajectory) / n_tok,
            "mean_head_entropy": sum(head_entropy_trajectory) / n_tok,
            "mean_head_gini": sum(gini_trajectory) / n_tok,

            # Per-layer profiles
            "avg_attn_per_layer": avg_attn_per_layer,
            "avg_mlp_per_layer": avg_mlp_per_layer,
            "avg_head_entropy_per_layer": avg_head_entropy_per_layer,

            # Per-head energy map
            "avg_head_energy": avg_head_energy,

            # Top active heads
            "top_10_heads": head_rankings[:10],

            # Trajectories for plotting
            "attn_mlp_trajectory": attn_mlp_trajectory,
            "head_entropy_trajectory": head_entropy_trajectory,
            "gini_trajectory": gini_trajectory,
            "risk_trajectory": risks,
        }

    # ── Lifecycle ────────────────────────────────────────────

    def feed_causal_primitives(self, cp) -> int:
        """
        Feed all recorded token diagnostics into a CausalPrimitives instance.

        Usage:
            from dflux import FineProbe
            from dflux.causal_primitives import CausalPrimitives

            probe = FineProbe.from_model(model)
            # ... run inference ...
            cp = CausalPrimitives(probe.n_layers, probe.n_heads)
            n = probe.feed_causal_primitives(cp)
            report = cp.compute()

        Returns the number of tokens fed.
        """
        n = 0
        for d in self.diagnostics:
            he = d.head_energies
            le = [sum(heads) for heads in he] if he else []
            cp.observe_token(he, le, d.J)
            n += 1
        return n

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset(self) -> None:
        """Reset for new generation."""
        self.diagnostics = []
        self.regime_history = []
        self.J = JState(window=self.cfg.window_tokens * 2)
        self._last_E_tail = None
        self._window_accum = 0.0
        self._window_start = 0
        self._forward_count = 0
        self._attn_seen = 0
        self._mlp_seen = 0
        self._layer_seen = 0
        self._layer_attn_norms = [None] * self.n_layers
        self._layer_mlp_norms = [None] * self.n_layers
        self._layer_output_norms = [None] * self.n_layers
        self._layer_head_norms = [None] * self.n_layers
