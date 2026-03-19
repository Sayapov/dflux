"""Standalone scale profile computation from telemetry diffs.

Compute, save, and load governor profiles without needing a model loaded.
"""

import json
from typing import Dict, List, Optional


def compute_scales(
    base_telemetry: dict,
    target_telemetry: dict,
    *,
    signal: str = "dilution_survival",
    strategy: str = "ratio",
    cap: float = 2.0,
    layer_type_bias: Optional[str] = None,
) -> Dict[int, float]:
    """Compute raw per-layer scale factors from two telemetry JSONs.

    Returns unblended scales (blend should be applied at use time).
    """
    base_agg = base_telemetry.get("aggregate", {})
    target_agg = target_telemetry.get("aggregate", {})
    base_vals = base_agg.get(f"{signal}_mean")
    target_vals = target_agg.get(f"{signal}_mean")

    if base_vals is None or target_vals is None:
        available = [k for k in base_agg if k.endswith("_mean")]
        raise ValueError(f"Signal '{signal}' not found. Available: {available}")

    n = min(len(base_vals), len(target_vals))
    layer_types = base_telemetry.get("layer_types")

    scales: Dict[int, float] = {}
    for i in range(n):
        if layer_type_bias is not None and layer_types:
            if i < len(layer_types) and layer_types[i] != layer_type_bias:
                scales[i] = 1.0
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

        scales[i] = s

    return scales


def blend_scales(scales: Dict[int, float], blend: float,
                 min_scale: float = 0.75, max_scale: float = 4.0) -> Dict[int, float]:
    """Apply blend interpolation: 0 = no effect, 1 = full target."""
    return {
        i: max(min_scale, min(max_scale, 1.0 + blend * (s - 1.0)))
        for i, s in scales.items()
    }


def compute_profile(
    base_telemetry: dict,
    target_telemetry: dict,
    *,
    signal: str = "dilution_survival",
    strategy: str = "ratio",
    blend: float = 0.3,
    cap: float = 2.0,
    layer_type_bias: Optional[str] = "full_attention",
) -> dict:
    """Compute a complete profile dict from two telemetry JSONs."""
    raw_scales = compute_scales(
        base_telemetry, target_telemetry,
        signal=signal, strategy=strategy, cap=cap,
        layer_type_bias=layer_type_bias,
    )
    blended = blend_scales(raw_scales, blend)

    # Only include non-1.0 scales for clarity
    active_scales = {
        str(i): round(s, 4) for i, s in blended.items() if abs(s - 1.0) > 1e-6
    }

    return {
        "signal": signal,
        "strategy": strategy,
        "blend": blend,
        "cap": cap,
        "layer_type_bias": layer_type_bias,
        "scales": active_scales,
    }


def save_profile(profile: dict, path: str) -> None:
    """Save a profile dict to JSON."""
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def load_profile(path: str) -> dict:
    """Load a profile dict from JSON. Converts string keys to int."""
    with open(path) as f:
        data = json.load(f)
    # Normalize scale keys to int
    if "scales" in data:
        data["scales"] = {int(k): v for k, v in data["scales"].items()}
    return data
