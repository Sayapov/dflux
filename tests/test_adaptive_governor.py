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


def test_adaptive_governor_config():
    """AdaptiveConfig can be instantiated with defaults."""
    from dflux.adaptive_governor import AdaptiveConfig

    cfg = AdaptiveConfig(
        window_size=32,
        ema_alpha=0.3,
        learning_rate=0.1,
        min_scale=0.75,
        max_scale=1.5,
    )
    assert cfg.window_size == 32
    assert cfg.ema_alpha == 0.3


def test_adaptive_governor_tick_logic():
    """Unit test the optimization pipeline: window → EMA → optimizer."""
    from dflux.adaptive_governor import (
        SignalWindow, EMATracker, ScaleOptimizer
    )

    n_layers = 3
    win_d = SignalWindow(size=4, n_layers=n_layers)
    win_e = SignalWindow(size=4, n_layers=n_layers)
    win_r = SignalWindow(size=4, n_layers=n_layers)

    for _ in range(4):
        win_d.push([0.5, 0.3, 0.7])
        win_e.push([0.2, 0.4, 0.1])
        win_r.push([1.0, 1.2, 0.8])

    assert win_d.is_full()

    ema_d = EMATracker(n_layers, alpha=0.3)
    ema_d.update(win_d.stats())

    opt = ScaleOptimizer(n_layers=n_layers, learning_rate=0.1)
    current = {0: 1.0, 1: 1.0, 2: 1.0}
    new = opt.step(current, {
        "dilution_mean": ema_d.mean,
        "entropy_mean": [0.2, 0.4, 0.1],
        "ratio_mean": [1.0, 1.2, 0.8],
    })

    assert any(abs(new[i] - 1.0) > 0.001 for i in range(n_layers))


def test_adaptive_governor_integration_gpt2():
    """Integration test: AdaptiveGovernor runs on a tiny model end-to-end."""
    pytest.importorskip("transformers")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflux.adaptive_governor import AdaptiveGovernor, AdaptiveConfig

    model_name = "gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception:
        pytest.skip("gpt2 not available")

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = AdaptiveConfig(window_size=4, learning_rate=0.1)
    gov = AdaptiveGovernor.signal_only(model, tokenizer, config=cfg)

    input_ids = tokenizer.encode("The quick brown fox", return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=16, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)

    report = gov.report()
    assert report["tokens_observed"] > 0
    assert report["windows_processed"] >= 1

    gov.detach()
