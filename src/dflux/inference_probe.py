#!/usr/bin/env python3
"""
Δ_flux Inference Probe — activation-space stability monitor.

Same fluid dynamics framework as training-time Δ_flux, but watches
activation norms instead of gradient norms. Detects:
  - Hallucination onset (energy migration into deep attention layers)
  - Reasoning degradation (turbulent activation flow)
  - Confidence collapse (regime transition during generation)

Works as a forward hook on any HuggingFace transformer. Captures
automatically during model.generate() — no manual calls needed.

Usage:
    from dflux.inference_probe import InferenceProbe

    probe = InferenceProbe.from_model(model)

    # Hooks fire automatically during generate
    output = model.generate(input_ids, max_new_tokens=256)

    report = probe.report()
    print(f"Mean hallucination risk: {report['mean_risk']:.3f}")
    print(f"Regime transitions: {len(report['regime_transitions'])}")

    # Per-token diagnostics
    for d in probe.diagnostics:
        print(f"  token {d.token_idx}: regime={d.regime}, risk={d.hallucination_risk:.3f}")

    # Clean up
    probe.detach()
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dflux.axe_ns import Regime, classify_regime, JState


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class ProbeConfig:
    """Configuration for the inference probe.

    Args:
        L_cut: Layer cutoff. Layers above this are the "tail" (deep layers).
               0 = auto (60% of total layers).
        window_tokens: Tokens per measurement window for flux accumulation.
        theta_warning: Flux threshold for regime alerts.
        j_func: Complexity functional.
                "tail_ratio" = E_tail / E_total (default, structural imbalance)
                "energy" = total activation energy
                "delta_ratio" = residual delta energy ratio
        track_deltas: Track per-layer residual deltas (layer contribution).
                      Stronger signal than raw norms — measures how much
                      each layer *changes* the representation.
        events_path: Optional path for JSONL event log.
    """
    L_cut: int = 0
    window_tokens: int = 16
    theta_warning: float = 0.3
    j_func: str = "tail_ratio"
    track_deltas: bool = True
    events_path: Optional[str] = None


@dataclass
class TokenDiagnostic:
    """Per-token structural diagnostic."""
    token_idx: int
    regime: str
    J: float
    E_tail: float
    E_total: float
    flux: float
    delta_flux: float               # Windowed accumulated flux
    hallucination_risk: float       # 0-1 composite score
    layer_norms: List[float] = field(default_factory=list)
    layer_deltas: Optional[List[float]] = None  # Per-layer residual contribution


# ══════════════════════════════════════════════════════════════
# INFERENCE PROBE
# ══════════════════════════════════════════════════════════════

class InferenceProbe:
    """Activation-space Δ_flux monitor for transformer inference.

    Hooks into transformer layers and measures activation energy
    distribution during forward passes. Captures automatically
    during model.generate() — each forward pass triggers hooks,
    and the probe processes one token per completed forward pass.

    Measures:
      - Activation L2 norms per layer (residual stream energy)
      - Residual deltas per layer (how much each layer changes the signal)
      - Regime classification from energy distribution
      - J trajectory (monotone complexity functional)
      - Hallucination risk composite score
    """

    def __init__(self, n_layers: int, cfg: ProbeConfig) -> None:
        self.n_layers = n_layers
        self.cfg = cfg
        if cfg.L_cut == 0:
            cfg.L_cut = max(0, int(0.6 * n_layers) - 1)

        # State
        self.J = JState(window=cfg.window_tokens * 2)
        self._last_E_tail: Optional[float] = None
        self._window_accum: float = 0.0
        self._window_start: int = 0
        self._token_count: int = 0
        self._forward_count: int = 0

        # Per-layer buffers populated by hooks
        self._layer_norms: List[Optional[float]] = [None] * n_layers
        self._layer_inputs: List[Optional[float]] = [None] * n_layers  # Input norms for delta
        self._layers_seen: int = 0

        # Hooks
        self._hooks: list = []
        self._model = None

        # Output
        self.diagnostics: List[TokenDiagnostic] = []
        self.regime_history: List[Regime] = []

        # Events file
        self._events_path: Optional[Path] = None
        if cfg.events_path:
            self._events_path = Path(cfg.events_path)
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            self._events_path.write_text("", encoding="utf-8")

    # ── Factory ──────────────────────────────────────────────

    @classmethod
    def from_model(
        cls,
        model,
        cfg: Optional[ProbeConfig] = None,
        **kwargs,
    ) -> "InferenceProbe":
        """Auto-detect transformer layers and attach hooks.

        Works with HuggingFace transformers:
          - GPT-2, GPT-J, GPT-Neo (model.transformer.h)
          - LLaMA, Mistral, Qwen, Phi (model.model.layers)
          - BERT, RoBERTa (model.encoder.layer)
          - Falcon (model.transformer.h)
          - MPT (model.transformer.blocks)
          - Generic ModuleList patterns
        """
        if cfg is None:
            cfg = ProbeConfig(**kwargs)

        layers = cls._find_transformer_layers(model)
        n_layers = len(layers)
        if n_layers == 0:
            raise ValueError(
                "Could not auto-detect transformer layers. "
                "Supported: GPT-2, LLaMA, Mistral, Qwen, BERT, Falcon, MPT. "
                "For custom architectures, pass the layer list manually."
            )

        probe = cls(n_layers=n_layers, cfg=cfg)
        probe._model = model
        probe._attach_hooks(layers)
        return probe

    @classmethod
    def from_layers(
        cls,
        layers: list,
        cfg: Optional[ProbeConfig] = None,
        **kwargs,
    ) -> "InferenceProbe":
        """Attach to an explicit list of nn.Module layers.

        Use this for custom architectures where auto-detection fails.
        """
        if cfg is None:
            cfg = ProbeConfig(**kwargs)

        probe = cls(n_layers=len(layers), cfg=cfg)
        probe._attach_hooks(layers)
        return probe

    @staticmethod
    def _find_transformer_layers(model) -> list:
        """Find the repeated transformer blocks in a model."""
        candidates = [
            "transformer.h",           # GPT-2, GPT-J, Falcon
            "model.layers",            # LLaMA, Mistral, Qwen, Phi
            "transformer.blocks",      # MPT
            "encoder.layer",           # BERT, RoBERTa
            "decoder.layers",          # T5 decoder
            "gpt_neox.layers",         # GPT-NeoX
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

        # Fallback: search for any ModuleList with >4 identical-type children
        import torch.nn as nn
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 4:
                types = [type(m).__name__ for m in module]
                if len(set(types)) == 1:  # All same type
                    return list(module)

        return []

    # ── Hook management ──────────────────────────────────────

    def _attach_hooks(self, layers: list) -> None:
        """Attach forward hooks that auto-capture and auto-process."""
        import torch

        self._hooks = []
        for i, layer in enumerate(layers):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # Extract hidden states from output
                    if isinstance(output, tuple):
                        hidden = output[0]
                    elif isinstance(output, dict):
                        hidden = output.get("hidden_states", output.get("last_hidden_state"))
                        if hidden is None:
                            return
                    else:
                        hidden = output

                    with torch.no_grad():
                        # hidden: [batch, seq_len, hidden_dim]
                        # During generation, seq_len=1 for cached steps
                        if hidden.dim() == 3:
                            h = hidden[:, -1, :]  # Last token position
                        elif hidden.dim() == 2:
                            h = hidden
                        else:
                            return

                        norm = float(h.norm().cpu())

                    # Store output norm
                    self._layer_norms[layer_idx] = norm

                    # Store input norm for delta computation
                    if self.cfg.track_deltas and input is not None:
                        inp = input[0] if isinstance(input, tuple) else input
                        if hasattr(inp, 'dim'):
                            with torch.no_grad():
                                if inp.dim() == 3:
                                    inp_h = inp[:, -1, :]
                                elif inp.dim() == 2:
                                    inp_h = inp
                                else:
                                    inp_h = None

                                if inp_h is not None:
                                    self._layer_inputs[layer_idx] = float(inp_h.norm().cpu())

                    self._layers_seen += 1

                    # When all layers have reported, process the token
                    if self._layers_seen >= self.n_layers:
                        self._auto_process()
                        self._layers_seen = 0

                return hook

            h = layer.register_forward_hook(make_hook(i))
            self._hooks.append(h)

    def _auto_process(self) -> None:
        """Called automatically when all layer hooks have fired for one forward pass."""
        token_idx = self._forward_count
        self._forward_count += 1

        norms = [n if n is not None else 0.0 for n in self._layer_norms]

        # Compute deltas (how much each layer changed the representation)
        deltas = None
        if self.cfg.track_deltas:
            deltas = []
            for i in range(self.n_layers):
                out_n = self._layer_norms[i] or 0.0
                in_n = self._layer_inputs[i] or 0.0
                # Delta = |output_norm - input_norm| / max(input_norm, eps)
                # Normalized so it's scale-independent
                deltas.append(abs(out_n - in_n) / max(in_n, 1e-12))

        # Energy computation
        E_total = sum(n ** 2 for n in norms)

        # Use deltas for energy if available and configured, else raw norms
        if deltas is not None and self.cfg.j_func == "delta_ratio":
            E_tail = sum(d ** 2 for d in deltas[self.cfg.L_cut + 1:])
            E_head = sum(d ** 2 for d in deltas[:self.cfg.L_cut + 1])
            E_total_j = E_head + E_tail
        else:
            E_tail = sum(n ** 2 for n in norms[self.cfg.L_cut + 1:])
            E_total_j = E_total

        # Flux
        if self._last_E_tail is None:
            flux = 0.0
        else:
            dE = E_tail - self._last_E_tail
            flux = max(dE, 0.0)
        self._last_E_tail = E_tail

        # Windowed accumulation
        self._window_accum += flux
        delta_flux = self._window_accum
        if (token_idx - self._window_start + 1) >= self.cfg.window_tokens:
            self._window_start = token_idx + 1
            self._window_accum = 0.0

        # Regime classification
        flux_ratio = flux / max(E_total_j, 1e-12)
        regime = classify_regime(E_tail, E_total_j, flux_ratio)
        self.regime_history.append(regime)

        # J functional
        if self.cfg.j_func == "tail_ratio" or self.cfg.j_func == "delta_ratio":
            j_val = E_tail / max(E_total_j, 1e-12)
        else:
            j_val = E_total_j
        self.J.record(j_val)

        # Hallucination risk
        risk = self._compute_hallucination_risk(regime, E_tail, E_total_j, delta_flux)

        diagnostic = TokenDiagnostic(
            token_idx=token_idx,
            regime=regime.value,
            J=j_val,
            E_tail=E_tail,
            E_total=E_total_j,
            flux=flux,
            delta_flux=delta_flux,
            hallucination_risk=risk,
            layer_norms=list(norms),
            layer_deltas=deltas,
        )
        self.diagnostics.append(diagnostic)
        self._token_count = token_idx

        # Emit event on regime transition or high risk
        if len(self.regime_history) >= 2 and self.regime_history[-1] != self.regime_history[-2]:
            self._emit_event(diagnostic, "regime_transition")
        if risk > 0.6:
            self._emit_event(diagnostic, "high_risk")

        # Reset buffers
        self._layer_norms = [None] * self.n_layers
        self._layer_inputs = [None] * self.n_layers

    def _compute_hallucination_risk(
        self,
        regime: Regime,
        E_tail: float,
        E_total: float,
        delta_flux: float,
    ) -> float:
        """Estimate hallucination risk from structural dynamics.

        Components:
          1. Regime severity (0-0.35) — turbulent/critical = higher risk
          2. Tail fraction (0-0.30) — deep layer energy concentration
          3. J trend (0-0.20) — positive trend = structural degradation
          4. Flux accumulation (0-0.15) — sustained energy migration
        """
        # Component 1: Regime severity
        regime_score = {
            Regime.LAMINAR: 0.0,
            Regime.TRANSITIONAL: 0.08,
            Regime.TURBULENT: 0.22,
            Regime.CRITICAL: 0.35,
        }.get(regime, 0.0)

        # Component 2: Tail fraction
        tail_frac = E_tail / max(E_total, 1e-12)
        tail_score = min(tail_frac, 1.0) * 0.30

        # Component 3: J trend (positive = degrading)
        trend = self.J.trend
        trend_score = min(max(trend * 50, 0.0), 0.20)

        # Component 4: Accumulated flux
        flux_score = min(delta_flux / max(self.cfg.theta_warning, 1e-12) * 0.15, 0.15)

        return min(regime_score + tail_score + trend_score + flux_score, 1.0)

    def _emit_event(self, d: TokenDiagnostic, event_type: str) -> None:
        """Write event to JSONL log."""
        if self._events_path is None:
            return
        event = {
            "schema_version": "2.0",
            "engine": "inference_probe",
            "event_type": event_type,
            "token_idx": d.token_idx,
            "regime": d.regime,
            "J": d.J,
            "E_tail": d.E_tail,
            "E_total": d.E_total,
            "flux": d.flux,
            "hallucination_risk": d.hallucination_risk,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    # ── Monitoring context manager ───────────────────────────

    def monitoring(self):
        """Context manager — resets state, captures during generate, done.

        Usage:
            with probe.monitoring():
                output = model.generate(input_ids, max_new_tokens=256)
            report = probe.report()
        """
        return _MonitorContext(self)

    # ── Report ───────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Generate summary report of the generation run."""
        if not self.diagnostics:
            return {"status": "no_data"}

        risks = [d.hallucination_risk for d in self.diagnostics]
        regimes = [d.regime for d in self.diagnostics]
        j_vals = [d.J for d in self.diagnostics]

        # Regime transitions
        transitions = []
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transitions.append({
                    "token": self.diagnostics[i].token_idx,
                    "from": regimes[i-1],
                    "to": regimes[i],
                    "risk": risks[i],
                })

        # High-risk spans
        high_risk_spans = []
        in_span = False
        span_start = 0
        for i, r in enumerate(risks):
            if r > 0.5 and not in_span:
                in_span = True
                span_start = i
            elif r <= 0.5 and in_span:
                in_span = False
                high_risk_spans.append({
                    "start_token": self.diagnostics[span_start].token_idx,
                    "end_token": self.diagnostics[i-1].token_idx,
                    "length": i - span_start,
                    "peak_risk": max(risks[span_start:i]),
                })
        if in_span:
            high_risk_spans.append({
                "start_token": self.diagnostics[span_start].token_idx,
                "end_token": self.diagnostics[-1].token_idx,
                "length": len(risks) - span_start,
                "peak_risk": max(risks[span_start:]),
            })

        # Regime distribution
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1

        # Layer energy heatmap data (avg norms per layer across all tokens)
        n_layers = self.n_layers
        layer_avg_norms = [0.0] * n_layers
        layer_avg_deltas = [0.0] * n_layers if self.cfg.track_deltas else None
        n_diag = len(self.diagnostics)
        for d in self.diagnostics:
            for i, n in enumerate(d.layer_norms):
                if i < n_layers:
                    layer_avg_norms[i] += n / n_diag
            if d.layer_deltas and layer_avg_deltas is not None:
                for i, delta in enumerate(d.layer_deltas):
                    if i < n_layers:
                        layer_avg_deltas[i] += delta / n_diag

        return {
            "total_tokens": len(self.diagnostics),
            "mean_risk": sum(risks) / len(risks),
            "max_risk": max(risks),
            "min_risk": min(risks),
            "std_risk": (sum((r - sum(risks)/len(risks))**2 for r in risks) / len(risks)) ** 0.5,
            "regime_distribution": regime_counts,
            "regime_transitions": transitions,
            "high_risk_spans": high_risk_spans,
            "J_final": self.J.current,
            "J_stabilized": self.J.stabilized,
            "J_trend": self.J.trend,
            "layer_avg_norms": layer_avg_norms,
            "layer_avg_deltas": layer_avg_deltas,
            "L_cut": self.cfg.L_cut,
        }

    # ── Lifecycle ────────────────────────────────────────────

    def detach(self) -> None:
        """Remove all hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset(self) -> None:
        """Reset state for a new generation."""
        self.diagnostics = []
        self.regime_history = []
        self.J = JState(window=self.cfg.window_tokens * 2)
        self._last_E_tail = None
        self._window_accum = 0.0
        self._window_start = 0
        self._forward_count = 0
        self._layers_seen = 0
        self._layer_norms = [None] * self.n_layers
        self._layer_inputs = [None] * self.n_layers


class _MonitorContext:
    """Context manager that resets probe state on entry."""
    def __init__(self, probe: InferenceProbe):
        self.probe = probe

    def __enter__(self):
        self.probe.reset()
        return self.probe

    def __exit__(self, *args):
        pass
