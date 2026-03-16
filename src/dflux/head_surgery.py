#!/usr/bin/env python3
"""
Head Surgery — direct structural intervention on attention head influence.

Uses the fine probe's energy fingerprint to identify which heads to boost
or dampen, then modifies their output projection matrices to change
their contribution to the residual stream.

This is NOT fine-tuning. No training, no gradients, no data. You're
directly adjusting the balance of power between the model's internal
functions based on measured structural telemetry.

Three intervention modes:
    1. Scale: multiply a head's output projection by a scalar
    2. Profile: apply a full intervention profile (boost some, dampen others)
    3. Auto: let the probe data determine the intervention automatically

Usage:
    from dflux.head_surgery import HeadSurgeon

    surgeon = HeadSurgeon(model)

    # Manual: boost the skeptic, dampen the fabricator
    surgeon.scale_head(layer=11, head=2, factor=2.0)   # 2x the skeptic
    surgeon.scale_head(layer=11, head=7, factor=0.7)   # dampen output head

    # Auto: use probe fingerprints to compute intervention
    surgeon.auto_calibrate(factual_report, halluc_report)

    # Undo everything
    surgeon.restore()
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class HeadIntervention:
    """Record of a single head modification."""
    layer: int
    head: int
    factor: float
    reason: str = ""


@dataclass
class SurgeryReport:
    """Summary of all interventions applied."""
    interventions: List[HeadIntervention] = field(default_factory=list)
    strategy: str = ""
    target_shift: str = ""

    def summary(self) -> str:
        lines = [f"Surgery Report ({self.strategy})"]
        lines.append(f"  Target: {self.target_shift}")
        lines.append(f"  Interventions: {len(self.interventions)}")
        for iv in self.interventions:
            direction = "BOOST" if iv.factor > 1.0 else "DAMPEN" if iv.factor < 1.0 else "NONE"
            lines.append(f"    L{iv.layer}H{iv.head}: {iv.factor:.2f}x ({direction}) — {iv.reason}")
        return "\n".join(lines)


class HeadSurgeon:
    """Direct structural intervention on attention heads.

    Modifies the output projection weights of specific attention heads
    to increase or decrease their contribution to the residual stream.

    Works with:
        GPT-2:   layer.attn.c_proj (Conv1D, weight shape [hidden, hidden])
        LLaMA:   layer.self_attn.o_proj (Linear, weight shape [hidden, hidden])
        Mistral: layer.self_attn.o_proj
        Qwen:    layer.self_attn.o_proj
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.layers = self._find_layers(model)
        self.n_layers = len(self.layers)
        self.n_heads = self._find_n_heads(model, self.layers[0])
        self.head_dim = self._find_head_dim(model)

        # Store original weights for restoration
        self._original_weights: Dict[Tuple[int, str], torch.Tensor] = {}
        self._interventions: List[HeadIntervention] = []

        # Detect output projection architecture
        self._proj_type = self._detect_proj_type(self.layers[0])

    @staticmethod
    def _find_layers(model) -> list:
        candidates = [
            "transformer.h", "model.layers", "transformer.blocks",
            "encoder.layer", "gpt_neox.layers", "layers",
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
        raise ValueError("Cannot find transformer layers")

    @staticmethod
    def _find_n_heads(model, sample_layer) -> int:
        if hasattr(model, 'config'):
            for attr in ['num_attention_heads', 'n_head', 'num_heads']:
                if hasattr(model.config, attr):
                    return getattr(model.config, attr)
        return 12

    @staticmethod
    def _find_head_dim(model) -> int:
        if hasattr(model, 'config'):
            hidden = getattr(model.config, 'hidden_size',
                             getattr(model.config, 'n_embd', 768))
            n_heads = getattr(model.config, 'num_attention_heads',
                              getattr(model.config, 'n_head', 12))
            return hidden // n_heads
        return 64

    @staticmethod
    def _detect_proj_type(layer) -> str:
        """Detect how the output projection is structured."""
        # GPT-2: layer.attn.c_proj (Conv1D)
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_proj'):
            return "gpt2"
        # LLaMA/Mistral/Qwen: layer.self_attn.o_proj (Linear)
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            return "llama"
        # GPT-NeoX/Pythia: layer.attention.dense (Linear)
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'dense'):
            return "neox"
        # Falcon: layer.self_attention.dense
        if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'dense'):
            return "falcon"
        # Fallback
        return "unknown"

    def _get_proj_module(self, layer_idx: int):
        """Get the output projection module for a layer."""
        layer = self.layers[layer_idx]
        if self._proj_type == "gpt2":
            return layer.attn.c_proj
        elif self._proj_type == "llama":
            return layer.self_attn.o_proj
        elif self._proj_type == "neox":
            return layer.attention.dense
        elif self._proj_type == "falcon":
            return layer.self_attention.dense
        else:
            raise ValueError(f"Unknown projection type: {self._proj_type}")

    # ── Core intervention ─────────────────────────────────────

    def scale_head(
        self,
        layer: int,
        head: int,
        factor: float,
        reason: str = "manual",
    ) -> None:
        """Scale a specific head's output projection.

        Args:
            layer: Layer index
            head: Head index within the layer
            factor: Multiplier (>1 = boost, <1 = dampen, 1 = no change)
            reason: Human-readable reason for the intervention
        """
        proj = self._get_proj_module(layer)
        key = (layer, "proj_weight")

        # Save original if first time touching this layer
        if key not in self._original_weights:
            self._original_weights[key] = proj.weight.data.clone()

        # Compute head slice indices
        # For GPT-2 Conv1D: weight shape is [in_features, out_features]
        # For Linear: weight shape is [out_features, in_features]
        start = head * self.head_dim
        end = start + self.head_dim

        with torch.no_grad():
            if self._proj_type == "gpt2":
                # Conv1D: weight is [in_features, out_features]
                # Each head's contribution is a slice of the input dimension
                # The output projection takes concatenated head outputs [h0|h1|...|hN]
                # and projects to hidden_dim. So we scale the INPUT slice.
                proj.weight.data[start:end, :] *= factor
            elif self._proj_type in ("llama", "neox", "falcon"):
                # Linear: weight is [out_features, in_features]
                # Same logic, but transposed
                proj.weight.data[:, start:end] *= factor

        self._interventions.append(HeadIntervention(
            layer=layer, head=head, factor=factor, reason=reason,
        ))

    def scale_heads_batch(
        self,
        interventions: List[Tuple[int, int, float, str]],
    ) -> None:
        """Apply multiple head scalings at once.

        Args:
            interventions: List of (layer, head, factor, reason) tuples
        """
        for layer, head, factor, reason in interventions:
            self.scale_head(layer, head, factor, reason)

    # ── Auto-calibration ──────────────────────────────────────

    def auto_calibrate(
        self,
        factual_report: Dict[str, Any],
        halluc_report: Dict[str, Any],
        boost_skeptics: float = 2.0,
        dampen_fabricators: float = 0.7,
        threshold_pct: float = 0.10,
    ) -> SurgeryReport:
        """Automatically compute and apply interventions from probe data.

        Compares factual vs hallucination head energy profiles.
        Boosts heads that are underactive during hallucination (skeptics).
        Dampens heads that are overactive during hallucination (fabricators).

        Args:
            factual_report: FineProbe report from factual prompts
            halluc_report: FineProbe report from hallucination prompts
            boost_skeptics: Factor to boost skeptic heads
            dampen_fabricators: Factor to dampen fabricator heads
            threshold_pct: Minimum relative change to trigger intervention
        """
        fact_heads = factual_report["avg_head_energy"]
        hall_heads = halluc_report["avg_head_energy"]
        n_layers = len(fact_heads)
        n_heads_per = len(fact_heads[0]) if n_layers > 0 else 0

        interventions = []

        for i in range(n_layers):
            for j in range(n_heads_per):
                f_energy = fact_heads[i][j]
                h_energy = hall_heads[i][j]

                if f_energy < 1e-6 and h_energy < 1e-6:
                    continue

                # Relative change
                base = max(f_energy, h_energy, 1e-12)
                rel_change = (h_energy - f_energy) / base

                if rel_change < -threshold_pct:
                    # Head is LESS active during hallucination = skeptic/suppression
                    # Boost it
                    factor = boost_skeptics
                    reason = (f"Skeptic: {rel_change:+.1%} during halluc "
                              f"(f={f_energy:.0f}, h={h_energy:.0f})")
                    interventions.append((i, j, factor, reason))

                elif rel_change > threshold_pct:
                    # Only dampen in the final layers where it matters
                    # Don't mess with early/mid layers
                    if i >= n_layers - 3:
                        # Check if this is a minor head with big relative spike
                        # (the arbitration heads we found)
                        avg_energy = (f_energy + h_energy) / 2
                        layer_total = sum(fact_heads[i]) + 1e-12
                        energy_share = avg_energy / layer_total

                        if energy_share < 0.05 and rel_change > 0.30:
                            # Minor head with big spike = arbitration overload
                            factor = dampen_fabricators
                            reason = (f"Arbitration overload: {rel_change:+.1%} during halluc, "
                                      f"only {energy_share:.1%} of layer energy")
                            interventions.append((i, j, factor, reason))

        # Apply all interventions
        self.scale_heads_batch(interventions)

        report = SurgeryReport(
            interventions=self._interventions.copy(),
            strategy="auto_calibrate",
            target_shift="Shift hallucination profile toward factual profile",
        )
        return report

    # ── Restoration ───────────────────────────────────────────

    def restore(self) -> None:
        """Undo all interventions, restore original weights."""
        for (layer_idx, key_name), original in self._original_weights.items():
            proj = self._get_proj_module(layer_idx)
            proj.weight.data.copy_(original)

        self._original_weights.clear()
        self._interventions.clear()

    # ── Inspection ────────────────────────────────────────────

    def get_head_norms(self, layer: int) -> List[float]:
        """Get current L2 norms of each head's output projection slice."""
        proj = self._get_proj_module(layer)
        norms = []
        with torch.no_grad():
            for h in range(self.n_heads):
                start = h * self.head_dim
                end = start + self.head_dim
                if self._proj_type == "gpt2":
                    slice_w = proj.weight.data[start:end, :]
                else:
                    slice_w = proj.weight.data[:, start:end]
                norms.append(float(slice_w.norm().cpu()))
        return norms

    def intervention_summary(self) -> str:
        """Human-readable summary of all interventions."""
        if not self._interventions:
            return "No interventions applied."
        report = SurgeryReport(
            interventions=self._interventions,
            strategy="manual" if len(self._interventions) < 5 else "auto",
        )
        return report.summary()
