"""
AdaptiveGovernor — windowed, EMA-tracked, multi-signal adaptive scaling.

Unlike the reactive LiveGovernor (which evaluates rules every token),
the AdaptiveGovernor:

  1. Collects signal windows over N tokens (default 32)
  2. Computes EMA-smoothed statistics per signal per layer
  3. Adjusts scales using gradient-free optimization toward a target profile
  4. Has hard triggers for pathological states (mode switches)

Signals used (hybrid):
  - dilution_survival: how much each layer's work persists to output
  - entropy_reduction: how much each layer compresses prediction entropy
  - mlp_attn_ratio: balance of MLP vs attention contribution

Usage:
    from dflux.adaptive_governor import AdaptiveGovernor

    gov = AdaptiveGovernor.from_profile(model, tokenizer, profile_path)
    output = model.generate(input_ids, max_new_tokens=256)
    gov.print_report()
    gov.detach()
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .multiscale_telemetry import MultiScaleTelemetry, TelemetryConfig, TokenSnapshot
from .live_governor import GovernorRule, GovernorIntervention


class SignalWindow:
    """Rolling window of per-layer signal values.

    Collects `size` snapshots of n-layer signals and computes
    windowed statistics (mean, std, trend) when full.
    """

    def __init__(self, size: int, n_layers: int) -> None:
        self.size = size
        self.n_layers = n_layers
        self._buf: deque[List[float]] = deque(maxlen=size)

    def push(self, values: List[float]) -> None:
        """Add one snapshot of per-layer values."""
        self._buf.append(list(values[:self.n_layers]))

    def is_full(self) -> bool:
        return len(self._buf) >= self.size

    def stats(self) -> Optional[Dict[str, List[float]]]:
        """Compute per-layer stats over the window.

        Returns None if window is not full.
        Returns dict with keys: mean, std, min, max, trend (per layer).
        Trend is slope of linear fit (positive = increasing).
        """
        if not self.is_full():
            return None

        n = len(self._buf)
        means = []
        stds = []
        mins = []
        maxs = []
        trends = []

        for layer_idx in range(self.n_layers):
            vals = [self._buf[t][layer_idx] for t in range(n)
                    if layer_idx < len(self._buf[t])]
            if not vals:
                means.append(0.0)
                stds.append(0.0)
                mins.append(0.0)
                maxs.append(0.0)
                trends.append(0.0)
                continue

            m = sum(vals) / len(vals)
            means.append(m)
            mins.append(min(vals))
            maxs.append(max(vals))

            if len(vals) > 1:
                var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                stds.append(math.sqrt(var))
                # Linear trend: slope of simple regression
                x_mean = (len(vals) - 1) / 2.0
                num = sum((t - x_mean) * (v - m) for t, v in enumerate(vals))
                den = sum((t - x_mean) ** 2 for t in range(len(vals)))
                trends.append(num / den if den > 0 else 0.0)
            else:
                stds.append(0.0)
                trends.append(0.0)

        return {
            "mean": means,
            "std": stds,
            "min": mins,
            "max": maxs,
            "trend": trends,
        }

    def clear(self) -> None:
        self._buf.clear()


# ── EMA Tracker ────────────────────────────────────────────────────

class EMATracker:
    """Exponential moving average of windowed signal statistics.

    Smooths out noise from individual windows to track the real
    trajectory of each signal per layer.

    Parameters
    ----------
    n_layers : int
        Number of layers to track.
    alpha : float
        EMA coefficient.  Higher = more weight on recent data.
        0.3 is a good default for 32-token windows.
    """

    def __init__(self, n_layers: int, alpha: float = 0.3) -> None:
        self.n_layers = n_layers
        self.alpha = alpha
        self.mean: List[float] = [0.0] * n_layers
        self.std: List[float] = [0.0] * n_layers
        self.trend: List[float] = [0.0] * n_layers
        self._initialized = False

    def update(self, stats: Dict[str, List[float]]) -> None:
        """Update EMA with new window statistics."""
        if not self._initialized:
            self.mean = list(stats["mean"])
            self.std = list(stats["std"])
            self.trend = list(stats.get("trend", [0.0] * self.n_layers))
            self._initialized = True
            return

        a = self.alpha
        for i in range(self.n_layers):
            if i < len(stats["mean"]):
                self.mean[i] = a * stats["mean"][i] + (1 - a) * self.mean[i]
            if i < len(stats["std"]):
                self.std[i] = a * stats["std"][i] + (1 - a) * self.std[i]
            if "trend" in stats and i < len(stats["trend"]):
                self.trend[i] = a * stats["trend"][i] + (1 - a) * self.trend[i]

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        self.mean = [0.0] * self.n_layers
        self.std = [0.0] * self.n_layers
        self.trend = [0.0] * self.n_layers
        self._initialized = False


# ── Scale Optimizer ────────────────────────────────────────────────

class ScaleOptimizer:
    """Gradient-free scale optimizer using hybrid signals.

    Computes a desired scale direction from three signals:
      - dilution_survival: low survival → boost (amplify layer's work)
      - entropy_reduction: high reduction → boost (layer is useful)
      - mlp_attn_ratio: used as stability indicator

    If a target profile is provided, the optimizer blends signal-driven
    adjustments with a compass pull toward the target scales.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    target_scales : dict[int, float] | None
        Target per-layer scale profile (compass). None = pure signal-driven.
    learning_rate : float
        How fast scales move per window (0.05-0.2 recommended).
    min_scale : float
        Lower clamp for scales.
    max_scale : float
        Upper clamp for scales.
    compass_weight : float
        How much to weight the target profile pull vs signal-driven (0-1).
        0.0 = pure signal, 1.0 = pure target following.
    signal_weights : dict
        Relative weights for each signal. Default: equal.
    """

    def __init__(
        self,
        n_layers: int,
        target_scales: Optional[Dict[int, float]] = None,
        learning_rate: float = 0.1,
        min_scale: float = 0.75,
        max_scale: float = 1.5,
        compass_weight: float = 0.4,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.n_layers = n_layers
        self.target_scales = target_scales or {}
        self.lr = learning_rate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.compass_weight = compass_weight
        self.signal_weights = signal_weights or {
            "dilution": 0.4,
            "entropy": 0.4,
            "ratio": 0.2,
        }

    def step(
        self,
        current_scales: Dict[int, float],
        signals: Dict[str, List[float]],
    ) -> Dict[int, float]:
        """Compute new scales from current scales and windowed signal stats.

        Parameters
        ----------
        current_scales : dict[int, float]
            Current per-layer scales.
        signals : dict
            Must contain keys: dilution_mean, entropy_mean, ratio_mean.
            Each value is a list of per-layer signal means.

        Returns
        -------
        dict[int, float]
            New per-layer scales, clamped to [min_scale, max_scale].
        """
        dilution = signals.get("dilution_mean", [0.5] * self.n_layers)
        entropy = signals.get("entropy_mean", [0.0] * self.n_layers)
        ratio = signals.get("ratio_mean", [1.0] * self.n_layers)

        # Normalize signals to [0, 1] range for combining
        def _normalize(vals):
            if not vals:
                return vals
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            if rng < 1e-10:
                return [0.5] * len(vals)
            return [(v - lo) / rng for v in vals]

        dilution_n = _normalize(dilution)
        entropy_n = _normalize(entropy)
        ratio_n = _normalize(ratio)

        new_scales = {}
        w = self.signal_weights

        for i in range(self.n_layers):
            current = current_scales.get(i, 1.0)

            # ── Signal-driven direction ──
            # Low dilution survival → layer's work gets wasted → boost it
            dilution_drive = 1.0 - dilution_n[i] if i < len(dilution_n) else 0.0
            # High entropy reduction → layer is doing useful compression → boost
            entropy_drive = entropy_n[i] if i < len(entropy_n) else 0.0
            # Ratio drive: center at 0.5 (neutral)
            ratio_drive = (ratio_n[i] - 0.5) if i < len(ratio_n) else 0.0

            signal_direction = (
                w["dilution"] * dilution_drive
                + w["entropy"] * entropy_drive
                + w["ratio"] * ratio_drive
            )
            # Map to [-1, 1] range: 0.5 is neutral
            signal_direction = (signal_direction - 0.3) * 2.0  # rough centering

            # ── Compass pull (toward target profile) ──
            compass_direction = 0.0
            if i in self.target_scales:
                target = self.target_scales[i]
                compass_direction = target - current

            # ── Blend signal + compass ──
            cw = self.compass_weight if self.target_scales else 0.0
            total_direction = (1.0 - cw) * signal_direction + cw * compass_direction

            # ── Apply with learning rate ──
            new = current + self.lr * total_direction
            new = max(self.min_scale, min(self.max_scale, new))
            new_scales[i] = new

        return new_scales


# ── Mode Triggers ──────────────────────────────────────────────────

@dataclass
class ModeTrigger:
    """Hard trigger for pathological states.

    When triggered, the governor switches to a protective scale profile
    instead of the normal optimization loop.

    Parameters
    ----------
    name : str
        Human-readable name for logging.
    signal : str
        Which signal to monitor (key in the signals dict).
    condition : str
        "mean_above" — mean of signal across layers exceeds threshold.
        "any_above_relative" — any layer exceeds threshold × mean.
        "trend_negative" — EMA trend is negative for most layers.
    threshold : float
        Trigger threshold.
    min_layers_triggered : int
        For "mean_above": how many layers must exceed threshold.
    protective_scales : dict
        Scale profile to apply when triggered. If None, scales reset to 1.0.
    cooldown : int
        Number of windows to stay in protective mode after trigger.
    """
    name: str
    signal: str
    condition: str = "mean_above"
    threshold: float = 1.0
    min_layers_triggered: int = 3
    protective_scales: Optional[Dict[int, float]] = None
    cooldown: int = 3
    _cooldown_remaining: int = field(default=0, init=False, repr=False)

    def check(self, signals: Dict[str, List[float]]) -> bool:
        """Check if trigger condition is met."""
        values = signals.get(self.signal)
        if values is None:
            return False

        if self.condition == "mean_above":
            count = sum(1 for v in values if v > self.threshold)
            return count >= self.min_layers_triggered

        elif self.condition == "any_above_relative":
            valid = [v for v in values if v is not None and math.isfinite(v)]
            if not valid:
                return False
            mean_val = sum(valid) / len(valid)
            if mean_val <= 0:
                return False
            return any(v > self.threshold * mean_val for v in valid)

        elif self.condition == "trend_negative":
            count = sum(1 for v in values if v < -abs(self.threshold))
            return count >= self.min_layers_triggered

        return False


# ── Adaptive Config ────────────────────────────────────────────────

@dataclass
class AdaptiveConfig:
    """Configuration for the AdaptiveGovernor."""
    window_size: int = 32
    ema_alpha: float = 0.3
    learning_rate: float = 0.1
    min_scale: float = 0.75
    max_scale: float = 1.5
    compass_weight: float = 0.4
    signal_weights: Optional[Dict[str, float]] = None
    enable_triggers: bool = True


# ── Adaptive Governor ──────────────────────────────────────────────

class AdaptiveGovernor:
    """Real-time adaptive governor with windowed signal optimization.

    Instead of evaluating rules every token (like LiveGovernor), this:
      1. Accumulates signals over a window of N tokens
      2. Computes EMA-smoothed statistics
      3. Runs gradient-free optimization to adjust per-layer scales
      4. Checks hard triggers for pathological states
    """

    SIGNALS = ("dilution_survival", "entropy_reduction", "mlp_attn_ratio")

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        config: Optional[AdaptiveConfig] = None,
        target_scales: Optional[Dict[int, float]] = None,
        triggers: Optional[List[ModeTrigger]] = None,
        telemetry_cfg: Optional[TelemetryConfig] = None,
    ) -> None:
        self.config = config or AdaptiveConfig()
        self.target_scales = target_scales

        # ── Telemetry ──
        if telemetry_cfg is None:
            telemetry_cfg = TelemetryConfig(
                logit_lens=True,
                logit_lens_top_k=5,
                cross_layer=False,
                mlp_internals=True,
                entropy_cascade=True,
                outlier_detection=False,
            )
        self.telem = MultiScaleTelemetry.from_model(model, tokenizer, cfg=telemetry_cfg)
        self.n_layers = self.telem.n_layers

        # Layer types
        self._transformer_layers = MultiScaleTelemetry._find_transformer_layers(model)
        self.layer_types: Optional[List[str]] = None
        if self._transformer_layers:
            self.layer_types = [
                MultiScaleTelemetry._get_layer_type(layer)
                for layer in self._transformer_layers
            ]

        # ── Scale hooks (actuator) ──
        self._scales: Dict[int, torch.Tensor] = {}
        self._scale_hooks: List = []
        self._install_scale_hooks(model)

        # ── Signal windows ──
        self._windows: Dict[str, SignalWindow] = {
            sig: SignalWindow(self.config.window_size, self.n_layers)
            for sig in self.SIGNALS
        }

        # ── EMA trackers ──
        self._emas: Dict[str, EMATracker] = {
            sig: EMATracker(self.n_layers, self.config.ema_alpha)
            for sig in self.SIGNALS
        }

        # ── Optimizer ──
        self._optimizer = ScaleOptimizer(
            n_layers=self.n_layers,
            target_scales=target_scales,
            learning_rate=self.config.learning_rate,
            min_scale=self.config.min_scale,
            max_scale=self.config.max_scale,
            compass_weight=self.config.compass_weight,
            signal_weights=self.config.signal_weights,
        )

        # ── Triggers ──
        self.triggers = triggers or self._default_triggers()
        self._in_protective_mode = False
        self._protective_cooldown = 0

        # ── Patch telemetry callback ──
        self._original_complete = self.telem._on_forward_complete
        self.telem._on_forward_complete = self._on_token_complete

        # ── Logging ──
        self._token_count = 0
        self._window_count = 0
        self.scale_history: List[Dict[int, float]] = []
        self.trigger_log: List[Dict] = []
        self.optimization_log: List[Dict] = []

    def _default_triggers(self) -> List[ModeTrigger]:
        """Default hard triggers based on experimental findings."""
        return [
            ModeTrigger(
                name="entropy_explosion",
                signal="entropy_mean",
                condition="mean_above",
                threshold=1.0,
                min_layers_triggered=3,
                protective_scales=None,
                cooldown=3,
            ),
            ModeTrigger(
                name="residual_flood",
                signal="dilution_mean",
                condition="any_above_relative",
                threshold=2.5,
                protective_scales=None,
                cooldown=2,
            ),
        ]

    def _install_scale_hooks(self, model: nn.Module) -> None:
        """Install mutable scale hooks on o_proj."""
        layers = self._transformer_layers
        device = next(model.parameters()).device

        for i, layer in enumerate(layers):
            scale = torch.ones(1, device=device, dtype=torch.float32)
            self._scales[i] = scale

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

            def _make_hook(s: torch.Tensor):
                def hook(module, input, output):
                    if s.item() == 1.0:
                        return output
                    return output * s.to(output.dtype)
                return hook

            h = o_proj.register_forward_hook(_make_hook(scale))
            self._scale_hooks.append(h)

    # ── Token callback ─────────────────────────────────────────────

    def _on_token_complete(self) -> None:
        """Fires after each token's telemetry snapshot."""
        self._original_complete()
        self._token_count += 1

        if not self.telem.snapshots:
            return
        snapshot = self.telem.snapshots[-1]

        # Push per-layer values into windows
        for sig_name in self.SIGNALS:
            values = getattr(snapshot, sig_name, None)
            if values is not None and isinstance(values, (list, tuple)):
                self._windows[sig_name].push(values)

        # Record current scales
        self.scale_history.append({
            i: self._scales[i].item() for i in range(self.n_layers)
        })

        # Check if all windows are full → time for optimization step
        all_full = all(w.is_full() for w in self._windows.values())
        if not all_full:
            return

        self._window_tick()

        # Clear windows for next cycle
        for w in self._windows.values():
            w.clear()

    def _window_tick(self) -> None:
        """Optimization step: runs every window_size tokens."""
        self._window_count += 1

        # Compute window stats and update EMAs
        for sig_name in self.SIGNALS:
            stats = self._windows[sig_name].stats()
            if stats is not None:
                self._emas[sig_name].update(stats)

        # Check hard triggers
        if self.config.enable_triggers and self._check_triggers():
            return

        # Protective cooldown
        if self._protective_cooldown > 0:
            self._protective_cooldown -= 1
            if self._protective_cooldown == 0:
                self._in_protective_mode = False
            return

        # Gradient-free optimization step
        current_scales = {i: self._scales[i].item() for i in range(self.n_layers)}

        signals = {}
        for sig_name, short in [("dilution_survival", "dilution"),
                                  ("entropy_reduction", "entropy"),
                                  ("mlp_attn_ratio", "ratio")]:
            ema = self._emas[sig_name]
            if ema.initialized:
                signals[f"{short}_mean"] = ema.mean
            else:
                signals[f"{short}_mean"] = [0.5] * self.n_layers

        new_scales = self._optimizer.step(current_scales, signals)

        for i, scale_val in new_scales.items():
            if i in self._scales:
                self._scales[i].fill_(scale_val)

        self.optimization_log.append({
            "window": self._window_count,
            "token": self._token_count,
            "scales": dict(new_scales),
            "signals": {k: list(v) for k, v in signals.items()},
        })

    def _check_triggers(self) -> bool:
        """Check all hard triggers. Returns True if any fired."""
        signals = {}
        for sig_name, short in [("dilution_survival", "dilution"),
                                  ("entropy_reduction", "entropy"),
                                  ("mlp_attn_ratio", "ratio")]:
            ema = self._emas[sig_name]
            if ema.initialized:
                signals[f"{short}_mean"] = ema.mean

        for trigger in self.triggers:
            if trigger.check(signals):
                self._fire_trigger(trigger)
                return True
        return False

    def _fire_trigger(self, trigger: ModeTrigger) -> None:
        """Apply protective scales from a triggered mode switch."""
        self._in_protective_mode = True
        self._protective_cooldown = trigger.cooldown

        if trigger.protective_scales:
            for i, scale_val in trigger.protective_scales.items():
                if i in self._scales:
                    self._scales[i].fill_(scale_val)
        else:
            for s in self._scales.values():
                s.fill_(1.0)

        self.trigger_log.append({
            "window": self._window_count,
            "token": self._token_count,
            "trigger": trigger.name,
            "cooldown": trigger.cooldown,
        })

    # ── Factory methods ────────────────────────────────────────────

    @classmethod
    def from_profile(
        cls,
        model: nn.Module,
        tokenizer: Any,
        profile_path: str,
        *,
        config: Optional[AdaptiveConfig] = None,
        **kwargs,
    ) -> "AdaptiveGovernor":
        """Create governor with target scales from a JSON profile."""
        from .profile import load_profile
        profile = load_profile(profile_path)
        target_scales = {int(k): float(v) for k, v in profile["scales"].items()}
        return cls(model, tokenizer, config=config, target_scales=target_scales, **kwargs)

    @classmethod
    def signal_only(
        cls,
        model: nn.Module,
        tokenizer: Any,
        *,
        config: Optional[AdaptiveConfig] = None,
        **kwargs,
    ) -> "AdaptiveGovernor":
        """Create governor with no target profile — pure signal-driven."""
        cfg = config or AdaptiveConfig()
        cfg.compass_weight = 0.0
        return cls(model, tokenizer, config=cfg, target_scales=None, **kwargs)

    # ── Reporting ──────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """Summary of what the governor did."""
        mean_scales = {}
        if self.scale_history:
            for i in range(self.n_layers):
                scales = [sh.get(i, 1.0) for sh in self.scale_history]
                mean_scales[i] = sum(scales) / len(scales)

        return {
            "tokens_observed": self._token_count,
            "windows_processed": self._window_count,
            "optimizations": len(self.optimization_log),
            "triggers_fired": len(self.trigger_log),
            "trigger_details": self.trigger_log,
            "mean_scales": mean_scales,
            "final_scales": {i: self._scales[i].item() for i in range(self.n_layers)},
            "in_protective_mode": self._in_protective_mode,
        }

    def print_report(self) -> None:
        """Print human-readable governor report."""
        r = self.report()
        print(f"\n{'='*60}")
        print(f"  ADAPTIVE GOVERNOR REPORT")
        print(f"{'='*60}")
        print(f"  Tokens observed:      {r['tokens_observed']}")
        print(f"  Windows processed:    {r['windows_processed']}")
        print(f"  Optimization steps:   {r['optimizations']}")
        print(f"  Triggers fired:       {r['triggers_fired']}")
        if r['trigger_details']:
            for t in r['trigger_details']:
                print(f"    ! {t['trigger']} at token {t['token']}")

        if r.get("mean_scales"):
            print(f"\n  -- Mean Scale Per Layer --")
            for i in sorted(r["mean_scales"].keys()):
                s = r["mean_scales"][i]
                if abs(s - 1.0) > 0.001:
                    lt = ""
                    if self.layer_types and i < len(self.layer_types):
                        lt = f" [{self.layer_types[i][:3]}]"
                    direction = "^" if s > 1.0 else "v"
                    print(f"  L{i:>2}{lt}: {s:.4f} {direction}")

        print(f"\n  -- Final Scales --")
        for i in sorted(r["final_scales"].keys()):
            s = r["final_scales"][i]
            if abs(s - 1.0) > 0.001:
                lt = ""
                if self.layer_types and i < len(self.layer_types):
                    lt = f" [{self.layer_types[i][:3]}]"
                direction = "^" if s > 1.0 else "v"
                print(f"  L{i:>2}{lt}: {s:.4f} {direction}")

    # ── Cleanup ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset scales, windows, EMAs — keep hooks installed."""
        for s in self._scales.values():
            s.fill_(1.0)
        for w in self._windows.values():
            w.clear()
        for e in self._emas.values():
            e.reset()
        self._token_count = 0
        self._window_count = 0
        self.scale_history.clear()
        self.trigger_log.clear()
        self.optimization_log.clear()
        self._in_protective_mode = False
        self._protective_cooldown = 0

    def detach(self) -> None:
        """Remove all hooks and restore telemetry callback."""
        for h in self._scale_hooks:
            h.remove()
        self._scale_hooks.clear()
        self.telem._on_forward_complete = self._original_complete
        self.telem.detach()
        for s in self._scales.values():
            s.fill_(1.0)
