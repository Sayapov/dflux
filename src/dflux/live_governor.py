"""
LiveGovernor — real-time inference-time head/layer scaling driven by telemetry.

The governor monitors telemetry signals during generation and adjusts
attention output scales between tokens.  Think of it as a PID controller
for transformer inference: telemetry is the sensor, rules are the
controller, and mutable scale hooks on `o_proj` are the actuator.

Design
------
1. Installs mutable scale tensors as forward hooks on each layer's
   attention output projection.  The hook multiplies the output by
   the current scale value — no weight mutation, fully reversible.

2. Patches the telemetry's `_on_forward_complete` so that after every
   token snapshot is captured the governor tick fires: rules are
   evaluated against the latest snapshot and scales are updated for
   the *next* token's forward pass.

3. Two modes:
   - **reactive** (default): scales reset to 1.0 each token, then
     rules add adjustments.  Stateless per-token.
   - **adaptive**: scales accumulate across tokens with exponential
     decay toward 1.0.  Tracks running behaviour.

Presets
-------
- `entropy_governor`  — amplify layers that reduce entropy, dampen
  layers that increase it.
- `dominance_damper`  — if one layer's attention norm exceeds N×
  the mean, dampen it.
- `survival_amplifier` — soft RYS: scale layers proportionally to
  their dilution survival.  High survival → amplify, low → dampen.

Usage
-----
    from dflux import LiveGovernor

    gov = LiveGovernor.survival_amplifier(model, tokenizer)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=64, do_sample=False)

    gov.report()       # see what the governor did
    gov.detach()       # remove all hooks
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .multiscale_telemetry import MultiScaleTelemetry, TelemetryConfig, TokenSnapshot


# ── Rule definition ─────────────────────────────────────────────────

@dataclass
class GovernorRule:
    """A single governance rule.

    Parameters
    ----------
    signal : str
        Name of a ``TokenSnapshot`` field (e.g. ``"entropy_reduction"``,
        ``"dilution_survival"``, ``"attn_norms"``).
    mode : str
        ``"threshold"`` — binary: if condition is met, apply *factor*.
        ``"proportional"`` — continuous: scale adjustment is
        ``factor * (value - mean) / range``.
    condition : str
        For threshold mode: ``"above"``, ``"below"``, or
        ``"relative_above"`` (value > threshold × layer-mean).
    threshold : float
        Trigger value (absolute or relative depending on *condition*).
    factor : float
        Threshold mode → multiply scale by this when triggered.
        Proportional mode → strength coefficient.
    layers : list[int] | None
        Restrict rule to these layer indices.  ``None`` = all layers.
    layer_type : str | None
        Restrict to ``"full_attention"`` or ``"linear_attention"``.
    """
    signal: str
    mode: str = "threshold"
    condition: str = "above"
    threshold: float = 0.0
    factor: float = 1.0
    layers: Optional[List[int]] = None
    layer_type: Optional[str] = None


@dataclass
class GovernorIntervention:
    """Record of a single scale adjustment."""
    token_idx: int
    layer: int
    rule_idx: int
    signal_value: float
    scale_before: float
    scale_after: float


# ── Governor ────────────────────────────────────────────────────────

class LiveGovernor:
    """Real-time inference governor driven by MultiScaleTelemetry."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        rules: List[GovernorRule],
        *,
        cfg: Optional[TelemetryConfig] = None,
        min_scale: float = 0.1,
        max_scale: float = 3.0,
        mode: str = "reactive",
        decay: float = 0.9,
    ) -> None:
        self.rules = list(rules)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode            # "reactive" | "adaptive"
        self.decay = decay          # only for adaptive mode

        # ── Telemetry (sensor) ──
        if cfg is None:
            cfg = TelemetryConfig(
                logit_lens=True,
                logit_lens_top_k=5,
                cross_layer=False,          # skip cross-layer sim (expensive)
                mlp_internals=True,
                entropy_cascade=True,
                outlier_detection=True,
            )
        self.telem = MultiScaleTelemetry.from_model(model, tokenizer, cfg=cfg)
        self.n_layers = self.telem.n_layers
        self.layer_types: Optional[List[str]] = None

        # Find transformer layers (same detection telemetry uses internally)
        self._transformer_layers = MultiScaleTelemetry._find_transformer_layers(model)

        # Detect layer types if available
        if self._transformer_layers:
            self.layer_types = [
                MultiScaleTelemetry._get_layer_type(layer)
                for layer in self._transformer_layers
            ]
        # Also pick up telemetry's own layer type detection if present
        if hasattr(self.telem, '_layer_types') and self.telem._layer_types:
            self.layer_types = list(self.telem._layer_types)

        # ── Mutable scale hooks (actuator) ──
        self._scales: Dict[int, torch.Tensor] = {}
        self._scale_hooks: List[torch.utils.hooks.RemovableHook] = []
        self._install_scale_hooks(model)

        # ── Patch telemetry callback ──
        self._original_complete = self.telem._on_forward_complete
        self.telem._on_forward_complete = self._patched_forward_complete

        # ── Logging ──
        self.interventions: List[GovernorIntervention] = []
        self.scale_history: List[Dict[int, float]] = []  # per-token

    # ── Hook installation ───────────────────────────────────────────

    def _install_scale_hooks(self, model: nn.Module) -> None:
        """Install forward hooks on each layer's attention output projection."""
        layers = self._transformer_layers
        device = next(model.parameters()).device

        for i, layer in enumerate(layers):
            scale = torch.ones(1, device=device, dtype=torch.float32)
            self._scales[i] = scale

            attn = MultiScaleTelemetry._find_attn_module(layer)
            if attn is None:
                continue

            # Find output projection
            o_proj = None
            for name in ("o_proj", "c_proj", "dense", "out_proj"):
                if hasattr(attn, name):
                    o_proj = getattr(attn, name)
                    break
            if o_proj is None:
                continue

            def _make_hook(s: torch.Tensor):
                def hook(module, input, output):
                    if s.item() == 1.0:
                        return output
                    return output * s.to(output.dtype)
                return hook

            h = o_proj.register_forward_hook(_make_hook(scale))
            self._scale_hooks.append(h)

    # ── Governor tick ───────────────────────────────────────────────

    def _patched_forward_complete(self) -> None:
        """Fires after telemetry captures a snapshot.  Evaluates rules."""
        self._original_complete()
        self._governor_tick()

    def _governor_tick(self) -> None:
        """Evaluate all rules against the latest snapshot, update scales."""
        if not self.telem.snapshots:
            return
        snapshot = self.telem.snapshots[-1]
        token_idx = snapshot.token_idx

        # Reactive: start from 1.0 each token
        # Adaptive: decay toward 1.0
        for i in range(self.n_layers):
            if self.mode == "reactive":
                self._scales[i].fill_(1.0)
            elif self.mode == "adaptive":
                current = self._scales[i].item()
                decayed = 1.0 + self.decay * (current - 1.0)
                self._scales[i].fill_(decayed)

        # Evaluate each rule
        for rule_idx, rule in enumerate(self.rules):
            values = getattr(snapshot, rule.signal, None)
            if values is None:
                continue
            if not isinstance(values, (list, tuple)):
                continue

            # Compute layer-mean for relative conditions
            valid_values = [v for v in values if v is not None and math.isfinite(v)]
            if not valid_values:
                continue
            mean_val = sum(valid_values) / len(valid_values)
            min_val = min(valid_values)
            max_val = max(valid_values)
            val_range = max_val - min_val + 1e-10

            for layer_idx in range(min(len(values), self.n_layers)):
                value = values[layer_idx]
                if value is None or not math.isfinite(value):
                    continue

                # Layer filters
                if rule.layers is not None and layer_idx not in rule.layers:
                    continue
                if rule.layer_type is not None and self.layer_types is not None:
                    if layer_idx < len(self.layer_types):
                        if self.layer_types[layer_idx] != rule.layer_type:
                            continue

                # ── Evaluate ──
                adjustment = 0.0

                if rule.mode == "threshold":
                    triggered = False
                    if rule.condition == "above" and value > rule.threshold:
                        triggered = True
                    elif rule.condition == "below" and value < rule.threshold:
                        triggered = True
                    elif rule.condition == "relative_above":
                        if mean_val > 0 and value > rule.threshold * mean_val:
                            triggered = True
                    elif rule.condition == "relative_below":
                        if mean_val > 0 and value < rule.threshold * mean_val:
                            triggered = True

                    if triggered:
                        adjustment = rule.factor

                elif rule.mode == "proportional":
                    # Continuous: scale = 1 + factor * (value - mean) / range
                    adjustment = 1.0 + rule.factor * (value - mean_val) / val_range

                # Apply adjustment
                if adjustment != 0.0:
                    old_scale = self._scales[layer_idx].item()

                    if rule.mode == "threshold":
                        new_scale = old_scale * adjustment
                    else:  # proportional
                        new_scale = old_scale * adjustment

                    new_scale = max(self.min_scale, min(self.max_scale, new_scale))
                    self._scales[layer_idx].fill_(new_scale)

                    if new_scale != old_scale:
                        self.interventions.append(GovernorIntervention(
                            token_idx=token_idx,
                            layer=layer_idx,
                            rule_idx=rule_idx,
                            signal_value=value,
                            scale_before=old_scale,
                            scale_after=new_scale,
                        ))

        # Record scale snapshot
        self.scale_history.append({
            i: self._scales[i].item() for i in range(self.n_layers)
        })

    # ── Presets ──────────────────────────────────────────────────────

    @classmethod
    def entropy_governor(
        cls,
        model: nn.Module,
        tokenizer: Any,
        amplify: float = 1.15,
        dampen: float = 0.85,
        **kwargs,
    ) -> "LiveGovernor":
        """Amplify layers that reduce entropy, dampen layers that increase it.

        Layers doing useful "thinking" (compressing prediction entropy)
        get boosted.  Layers that add confusion get dampened.
        """
        rules = [
            GovernorRule(
                signal="entropy_reduction",
                mode="threshold",
                condition="above",
                threshold=0.1,      # meaningful entropy reduction
                factor=amplify,
            ),
            GovernorRule(
                signal="entropy_reduction",
                mode="threshold",
                condition="below",
                threshold=-0.05,    # entropy went UP (bad)
                factor=dampen,
            ),
        ]
        return cls(model, tokenizer, rules, **kwargs)

    @classmethod
    def dominance_damper(
        cls,
        model: nn.Module,
        tokenizer: Any,
        threshold_ratio: float = 2.0,
        dampen: float = 0.8,
        **kwargs,
    ) -> "LiveGovernor":
        """Dampen layers whose attention norm exceeds threshold × mean.

        Prevents any single layer from hijacking the residual stream.
        """
        rules = [
            GovernorRule(
                signal="attn_norms",
                mode="threshold",
                condition="relative_above",
                threshold=threshold_ratio,
                factor=dampen,
            ),
        ]
        return cls(model, tokenizer, rules, **kwargs)

    @classmethod
    def survival_amplifier(
        cls,
        model: nn.Module,
        tokenizer: Any,
        strength: float = 0.5,
        **kwargs,
    ) -> "LiveGovernor":
        """Soft RYS: scale layers proportionally to their dilution survival.

        Layers whose work survives to the final output get amplified.
        Layers whose work gets diluted away get dampened.
        This is a continuous, per-token version of RYS's layer
        duplication — instead of copying high-value circuits, we
        amplify them dynamically.
        """
        rules = [
            GovernorRule(
                signal="dilution_survival",
                mode="proportional",
                factor=strength,
            ),
        ]
        return cls(model, tokenizer, rules, **kwargs)

    @classmethod
    def hybrid_governor(
        cls,
        model: nn.Module,
        tokenizer: Any,
        entropy_amplify: float = 1.12,
        entropy_dampen: float = 0.88,
        survival_strength: float = 0.3,
        dominance_threshold: float = 2.5,
        dominance_dampen: float = 0.85,
        **kwargs,
    ) -> "LiveGovernor":
        """Combined governor: entropy + survival + dominance."""
        rules = [
            # Entropy: reward useful layers, penalize confusing ones
            GovernorRule(
                signal="entropy_reduction",
                mode="threshold",
                condition="above",
                threshold=0.1,
                factor=entropy_amplify,
            ),
            GovernorRule(
                signal="entropy_reduction",
                mode="threshold",
                condition="below",
                threshold=-0.05,
                factor=entropy_dampen,
            ),
            # Survival: proportional scaling
            GovernorRule(
                signal="dilution_survival",
                mode="proportional",
                factor=survival_strength,
            ),
            # Dominance: prevent runaway layers
            GovernorRule(
                signal="attn_norms",
                mode="threshold",
                condition="relative_above",
                threshold=dominance_threshold,
                factor=dominance_dampen,
            ),
        ]
        return cls(model, tokenizer, rules, **kwargs)

    @classmethod
    def distillation_governor(
        cls,
        model: nn.Module,
        tokenizer: Any,
        target_scales: Dict[int, float],
        *,
        blend: float = 1.0,
        **kwargs,
    ) -> "LiveGovernor":
        """Apply a static per-layer scale profile derived from telemetry diff.

        Instead of reactive rules, this preset applies fixed scales that
        push the base model's layer dynamics toward a target model's profile.
        The scales are pre-computed from telemetry comparison and applied
        every token (no signal evaluation needed).

        Parameters
        ----------
        target_scales : dict[int, float]
            Per-layer scale factors.  E.g. ``{3: 1.25, 7: 1.25, ...}``
            for full attention layer boost.
        blend : float
            Interpolation: 0.0 = no effect, 1.0 = full target scales.
            Values > 1.0 overshoot (amplify the diff).
        """
        # Use an empty rule list — we override scales directly in __init__
        gov = cls(model, tokenizer, rules=[], **kwargs)

        # Apply target scales (blended toward 1.0)
        for layer_idx, target in target_scales.items():
            if layer_idx in gov._scales:
                blended = 1.0 + blend * (target - 1.0)
                blended = max(gov.min_scale, min(gov.max_scale, blended))
                gov._scales[layer_idx].fill_(blended)

        # Override the tick to just hold static scales (no rule eval)
        def _static_tick(self_ref=gov):
            """Static governor: record scales but don't change them."""
            if self_ref.telem.snapshots:
                self_ref.scale_history.append({
                    i: self_ref._scales[i].item()
                    for i in range(self_ref.n_layers)
                })
        gov._governor_tick = _static_tick

        return gov

    @classmethod
    def from_telemetry_diff(
        cls,
        model: nn.Module,
        tokenizer: Any,
        base_telemetry: dict,
        target_telemetry: dict,
        *,
        signal: str = "entropy_reduction",
        strategy: str = "ratio",
        blend: float = 1.0,
        cap: float = 2.0,
        layer_type_bias: Optional[str] = None,
        **kwargs,
    ) -> "LiveGovernor":
        """Compute per-layer scales from telemetry diff, then apply them.

        This is the "inference-time distillation" preset.  Given telemetry
        from a base model and a target (e.g. reasoning-distilled) model,
        compute how much each layer should be scaled to shift the base
        toward the target's profile.

        Parameters
        ----------
        base_telemetry : dict
            Telemetry JSON from the base model (``telem.save()`` output).
        target_telemetry : dict
            Telemetry JSON from the target model.
        signal : str
            Which signal to use for computing scales.  Good choices:
            ``"entropy_reduction"`` (how much each layer compresses),
            ``"dilution_survival"`` (how much each layer's work persists),
            ``"attn_norms"`` (attention loudness).
        strategy : str
            ``"ratio"`` — scale[i] = target[i] / base[i].
            ``"delta"`` — scale[i] = 1 + (target[i] - base[i]) / range.
        blend : float
            0 = no effect, 1 = match target, >1 = overshoot.
        cap : float
            Maximum absolute scale value (safety clamp).
        layer_type_bias : str or None
            If set to ``"full_attention"`` or ``"linear_attention"``,
            only apply scales to that layer type, leave others at 1.0.
        """
        base_agg = base_telemetry.get("aggregate", {})
        target_agg = target_telemetry.get("aggregate", {})
        base_vals = base_agg.get(f"{signal}_mean")
        target_vals = target_agg.get(f"{signal}_mean")

        if base_vals is None or target_vals is None:
            raise ValueError(f"Signal '{signal}' not found in telemetry data. "
                             f"Available: {[k for k in base_agg if k.endswith('_mean')]}")

        n = min(len(base_vals), len(target_vals))
        layer_types = base_telemetry.get("layer_types")

        target_scales: Dict[int, float] = {}
        for i in range(n):
            # Layer type filter
            if layer_type_bias is not None and layer_types:
                if i < len(layer_types) and layer_types[i] != layer_type_bias:
                    target_scales[i] = 1.0
                    continue

            bv = base_vals[i]
            tv = target_vals[i]

            if strategy == "ratio":
                if abs(bv) < 1e-10:
                    s = 1.0
                else:
                    s = tv / bv
                    s = max(1.0 / cap, min(cap, s))
            elif strategy == "delta":
                val_range = max(abs(max(base_vals)), abs(min(base_vals)), 1e-10)
                s = 1.0 + (tv - bv) / val_range
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            target_scales[i] = s

        return cls.distillation_governor(
            model, tokenizer, target_scales, blend=blend, **kwargs
        )

    # ── Reporting ────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Summary of what the governor did."""
        if not self.interventions:
            return {
                "total_interventions": 0,
                "tokens_observed": len(self.scale_history),
                "verdict": "No rules triggered",
            }

        # Count per layer
        layer_counts: Dict[int, int] = {}
        layer_mean_scale: Dict[int, List[float]] = {}
        for iv in self.interventions:
            layer_counts[iv.layer] = layer_counts.get(iv.layer, 0) + 1
            if iv.layer not in layer_mean_scale:
                layer_mean_scale[iv.layer] = []
            layer_mean_scale[iv.layer].append(iv.scale_after)

        # Count per rule
        rule_counts: Dict[int, int] = {}
        for iv in self.interventions:
            rule_counts[iv.rule_idx] = rule_counts.get(iv.rule_idx, 0) + 1

        # Most active layers
        top_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Mean scale per layer over entire generation
        mean_scales = {}
        if self.scale_history:
            for i in range(self.n_layers):
                scales = [sh.get(i, 1.0) for sh in self.scale_history]
                mean_scales[i] = sum(scales) / len(scales)

        return {
            "total_interventions": len(self.interventions),
            "tokens_observed": len(self.scale_history),
            "interventions_per_token": len(self.interventions) / max(len(self.scale_history), 1),
            "top_layers": [
                {"layer": l, "count": c,
                 "mean_scale": sum(layer_mean_scale[l]) / len(layer_mean_scale[l]),
                 "layer_type": self.layer_types[l] if self.layer_types and l < len(self.layer_types) else "unknown"}
                for l, c in top_layers
            ],
            "rule_activity": {
                i: {"rule": f"{self.rules[i].signal} {self.rules[i].condition} {self.rules[i].threshold}",
                    "triggers": c}
                for i, c in sorted(rule_counts.items())
            },
            "mean_scales": mean_scales,
        }

    def print_report(self) -> None:
        """Print human-readable governor report."""
        r = self.report()
        print(f"\n{'='*60}")
        print(f"  LIVE GOVERNOR REPORT")
        print(f"{'='*60}")
        print(f"  Tokens observed:  {r['tokens_observed']}")
        print(f"  Total interventions: {r['total_interventions']}")
        print(f"  Per token: {r.get('interventions_per_token', 0):.1f}")

        if r.get("top_layers"):
            print(f"\n  ── Most Active Layers ──")
            for entry in r["top_layers"]:
                lt = entry.get("layer_type", "?")[:3]
                bar = "█" * min(40, entry["count"])
                print(f"  L{entry['layer']:>2} [{lt}]: {entry['count']:>4} triggers, "
                      f"mean_scale={entry['mean_scale']:.3f}  {bar}")

        if r.get("rule_activity"):
            print(f"\n  ── Rule Activity ──")
            for idx, info in r["rule_activity"].items():
                print(f"  Rule {idx}: {info['rule']} → {info['triggers']} triggers")

        if r.get("mean_scales"):
            print(f"\n  ── Mean Scale Per Layer ──")
            scales = r["mean_scales"]
            max_dev = max(abs(s - 1.0) for s in scales.values()) if scales else 0
            if max_dev < 0.001:
                print(f"  All layers at 1.000 (no adjustments applied)")
            else:
                for i in sorted(scales.keys()):
                    s = scales[i]
                    if abs(s - 1.0) > 0.001:
                        lt = ""
                        if self.layer_types and i < len(self.layer_types):
                            lt = f" [{self.layer_types[i][:3]}]"
                        direction = "↑" if s > 1.0 else "↓"
                        print(f"  L{i:>2}{lt}: {s:.3f} {direction}")

    # ── Cleanup ──────────────────────────────────────────────────────

    def reset_scales(self) -> None:
        """Reset all scales to 1.0."""
        for s in self._scales.values():
            s.fill_(1.0)

    def detach(self) -> None:
        """Remove all governor hooks and restore telemetry callback."""
        for h in self._scale_hooks:
            h.remove()
        self._scale_hooks.clear()
        self.telem._on_forward_complete = self._original_complete
        self.telem.detach()
        self.reset_scales()
