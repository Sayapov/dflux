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
