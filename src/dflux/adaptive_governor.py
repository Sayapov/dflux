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
