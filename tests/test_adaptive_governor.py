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


def test_scale_optimizer_nudges_toward_target():
    """ScaleOptimizer nudges current scales toward target based on signals."""
    from dflux.adaptive_governor import ScaleOptimizer

    opt = ScaleOptimizer(
        n_layers=3,
        target_scales={0: 1.2, 1: 0.9, 2: 1.0},
        learning_rate=0.1,
        min_scale=0.5,
        max_scale=2.0,
    )

    current = {0: 1.0, 1: 1.0, 2: 1.0}
    signals = {
        "dilution_mean": [0.3, 0.7, 0.5],
        "entropy_mean": [0.5, 0.3, 0.4],
        "ratio_mean": [1.0, 1.0, 1.0],
    }

    new_scales = opt.step(current, signals)
    # Layer 0 should move toward 1.2 (increase)
    assert new_scales[0] > 1.0
    # Layer 1 should move toward 0.9 (decrease)
    assert new_scales[1] < 1.0
    # All within bounds
    for s in new_scales.values():
        assert 0.5 <= s <= 2.0


def test_scale_optimizer_respects_bounds():
    """ScaleOptimizer clamps to min/max."""
    from dflux.adaptive_governor import ScaleOptimizer

    opt = ScaleOptimizer(
        n_layers=1,
        target_scales={0: 5.0},
        learning_rate=1.0,
        min_scale=0.75,
        max_scale=1.5,
    )

    current = {0: 1.4}
    signals = {"dilution_mean": [0.1], "entropy_mean": [0.5], "ratio_mean": [1.0]}
    new_scales = opt.step(current, signals)
    assert new_scales[0] <= 1.5


def test_scale_optimizer_no_target_stays_signal_driven():
    """Without target profile, optimizer is purely signal-driven."""
    from dflux.adaptive_governor import ScaleOptimizer

    opt = ScaleOptimizer(
        n_layers=2,
        target_scales=None,
        learning_rate=0.1,
        min_scale=0.75,
        max_scale=1.5,
    )

    current = {0: 1.0, 1: 1.0}
    signals = {
        "dilution_mean": [0.2, 0.8],
        "entropy_mean": [0.5, 0.5],
        "ratio_mean": [1.0, 1.0],
    }
    new_scales = opt.step(current, signals)
    # Low dilution = low survival = needs amplification
    assert new_scales[0] > 1.0


def test_mode_trigger_detects_entropy_explosion():
    """ModeTrigger fires when entropy exceeds threshold across layers."""
    from dflux.adaptive_governor import ModeTrigger

    trigger = ModeTrigger(
        name="entropy_explosion",
        signal="entropy_mean",
        condition="mean_above",
        threshold=1.0,
        min_layers_triggered=3,
    )

    signals = {"entropy_mean": [0.5, 0.6, 0.7, 0.8]}
    assert not trigger.check(signals)

    signals = {"entropy_mean": [1.2, 1.5, 0.8, 1.3]}
    assert trigger.check(signals)


def test_mode_trigger_detects_residual_flood():
    """ModeTrigger fires when one layer's signal is N× the mean."""
    from dflux.adaptive_governor import ModeTrigger

    trigger = ModeTrigger(
        name="residual_flood",
        signal="dilution_mean",
        condition="any_above_relative",
        threshold=2.5,
    )

    signals = {"dilution_mean": [0.5, 0.6, 0.55, 0.5]}
    assert not trigger.check(signals)

    signals = {"dilution_mean": [0.3, 0.3, 2.0, 0.3]}
    assert trigger.check(signals)


def test_mode_trigger_returns_protective_profile():
    """When triggered, ModeTrigger provides a protective scale profile."""
    from dflux.adaptive_governor import ModeTrigger

    trigger = ModeTrigger(
        name="test",
        signal="entropy_mean",
        condition="mean_above",
        threshold=1.0,
        protective_scales={3: 0.8, 7: 0.8},
    )

    assert trigger.protective_scales == {3: 0.8, 7: 0.8}
