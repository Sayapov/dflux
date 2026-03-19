"""Tests for AdaptiveGovernor signal window and scale logic."""

import pytest
import math


def test_signal_window_accumulates():
    """SignalWindow collects values and computes stats over a window."""
    from dflux.adaptive_governor import SignalWindow

    win = SignalWindow(size=4, n_layers=3)
    win.push([1.0, 2.0, 3.0])
    win.push([1.5, 2.5, 3.5])
    win.push([1.2, 2.2, 3.2])
    win.push([1.8, 2.8, 3.8])

    assert win.is_full()
    stats = win.stats()
    assert abs(stats["mean"][0] - 1.375) < 1e-6
    assert abs(stats["mean"][2] - 3.375) < 1e-6
    assert stats["std"][0] > 0
    assert "trend" in stats


def test_signal_window_rolls():
    """Window drops oldest values when full."""
    from dflux.adaptive_governor import SignalWindow

    win = SignalWindow(size=2, n_layers=2)
    win.push([1.0, 2.0])
    win.push([3.0, 4.0])
    assert win.is_full()

    win.push([5.0, 6.0])
    stats = win.stats()
    assert abs(stats["mean"][0] - 4.0) < 1e-6


def test_signal_window_not_full():
    """Stats return None when window is not full."""
    from dflux.adaptive_governor import SignalWindow

    win = SignalWindow(size=4, n_layers=2)
    win.push([1.0, 2.0])
    assert not win.is_full()
    assert win.stats() is None


def test_ema_tracker_smooths():
    """EMATracker applies exponential moving average to signal stats."""
    from dflux.adaptive_governor import EMATracker

    ema = EMATracker(n_layers=2, alpha=0.5)
    ema.update({"mean": [1.0, 2.0], "std": [0.1, 0.2], "trend": [0.0, 0.0],
                "min": [0.9, 1.8], "max": [1.1, 2.2]})
    assert abs(ema.mean[0] - 1.0) < 1e-6

    ema.update({"mean": [3.0, 4.0], "std": [0.3, 0.4], "trend": [0.1, 0.1],
                "min": [2.8, 3.8], "max": [3.2, 4.2]})
    assert abs(ema.mean[0] - 2.0) < 1e-6
    assert abs(ema.mean[1] - 3.0) < 1e-6


def test_ema_tracker_trend():
    """EMATracker tracks smoothed trend."""
    from dflux.adaptive_governor import EMATracker

    ema = EMATracker(n_layers=1, alpha=1.0)
    ema.update({"mean": [5.0], "std": [0.1], "trend": [0.5],
                "min": [4.5], "max": [5.5]})
    assert abs(ema.trend[0] - 0.5) < 1e-6
