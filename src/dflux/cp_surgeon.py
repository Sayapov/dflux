#!/usr/bin/env python3
"""
CP-Guided Head Surgery — information-theoretic auto-calibration.

Closes the loop between CausalPrimitives (diagnosis) and HeadSurgeon
(intervention). Instead of heuristic "boost skeptics, dampen fabricators",
we use the Jansma & Hoel criterion: maximize emergent complexity
S_path × S_row_bar by redistributing causal load across heads.

The target distribution isn't arbitrary — it comes from the model's own
training dynamics. Early in pretraining, models discover distributed
causal structure (high CP across many heads), then gradient descent
destroys it by concentrating everything in a few dominant heads.
CP-guided surgery restores what training killed.

Pipeline:
    1. Measure current CP distribution
    2. Compute target scales that maximize S_path × S_row_bar
    3. Apply via HeadSurgeon (milliseconds)
    4. Re-measure CP to verify improvement
    5. Iterate until convergence

Usage:
    from dflux import FineProbe, HeadSurgeon
    from dflux.cp_surgeon import CPSurgeon

    probe = FineProbe.from_model(model)
    surgeon = HeadSurgeon(model)
    cp_surgeon = CPSurgeon(probe, surgeon)

    # Auto-tune: measure → compute → apply → verify
    result = cp_surgeon.auto_tune(model, tokenizer, prompts)

    # result.before_emergence vs result.after_emergence
    # result.interventions — what was changed
    # result.convergence_history — emergence at each iteration

Free instrument. MIT license.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from dflux.causal_primitives import CausalPrimitives, CPConfig
from dflux.fine_probe import FineProbe
from dflux.head_surgery import HeadSurgeon, HeadIntervention, SurgeryReport


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class CPSurgeonConfig:
    """Configuration for CP-guided surgery."""

    # Target distribution parameters
    target_mode: str = "maximize_emergence"
    # "maximize_emergence" — optimize S_path × S_row_bar
    # "equalize_cp" — push all heads toward mean CP
    # "restore_early" — if you have early checkpoint data, restore that distribution

    # Surgery parameters
    max_boost: float = 2.0       # Maximum scale factor for underpowered heads
    min_dampen: float = 0.3      # Minimum scale factor for overpowered heads
    dead_head_threshold: float = 0.02   # CP below this = head is causally dead
    dominant_threshold: float = 0.8     # Head with this fraction of layer CP = dominant

    # Convergence
    max_iterations: int = 5
    convergence_threshold: float = 0.001  # Stop when emergence change < this
    learning_rate: float = 0.5   # How aggressively to apply corrections (0-1)

    # Measurement
    n_bins: int = 16
    max_tokens: int = 48
    temperature: float = 0.8


# ═══════════════════════════════════════════════════════════
# SURGERY RESULT
# ═══════════════════════════════════════════════════════════

@dataclass
class CPSurgeryResult:
    """Full result of CP-guided surgery."""

    # Before/after
    before_cp: Dict[str, Any] = field(default_factory=dict)
    after_cp: Dict[str, Any] = field(default_factory=dict)
    before_emergence: float = 0.0
    after_emergence: float = 0.0
    emergence_improvement: float = 0.0

    # Hierarchy shift
    before_hierarchy: str = ""
    after_hierarchy: str = ""

    # Convergence
    convergence_history: List[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False

    # What was done
    interventions: List[HeadIntervention] = field(default_factory=list)
    head_scales: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Top head changes
    before_top_heads: List[Dict] = field(default_factory=list)
    after_top_heads: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "CP-Guided Surgery Result",
            "=" * 50,
            f"Iterations:  {self.iterations} (converged: {self.converged})",
            f"Emergence:   {self.before_emergence:.6f} → {self.after_emergence:.6f} "
            f"({self.emergence_improvement:+.1%})",
            f"Hierarchy:   {self.before_hierarchy} → {self.after_hierarchy}",
            f"Interventions: {len(self.interventions)}",
            "",
            "Convergence history:",
        ]
        for i, e in enumerate(self.convergence_history):
            bar = "█" * int(e * 10000)
            lines.append(f"  iter {i}: {e:.6f} {bar}")

        lines.append("")
        lines.append("Top 5 before → after:")
        for i in range(min(5, len(self.before_top_heads))):
            b = self.before_top_heads[i] if i < len(self.before_top_heads) else {}
            a = self.after_top_heads[i] if i < len(self.after_top_heads) else {}
            lines.append(
                f"  #{i+1}: L{b.get('layer','?')}H{b.get('head','?')}"
                f"(CP={b.get('cp',0):.4f})"
                f" → L{a.get('layer','?')}H{a.get('head','?')}"
                f"(CP={a.get('cp',0):.4f})"
            )

        # Show biggest interventions
        sorted_iv = sorted(self.interventions,
                           key=lambda iv: abs(iv.factor - 1.0), reverse=True)
        if sorted_iv:
            lines.append("")
            lines.append("Biggest interventions:")
            for iv in sorted_iv[:10]:
                direction = "BOOST" if iv.factor > 1.0 else "DAMPEN"
                lines.append(f"  L{iv.layer}H{iv.head}: {iv.factor:.3f}x ({direction}) — {iv.reason}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# CP SURGEON
# ═══════════════════════════════════════════════════════════

class CPSurgeon:
    """
    Information-theoretic auto-calibration of attention heads.

    Measures causal primitives, computes optimal per-head scaling
    to maximize emergent complexity, applies via HeadSurgeon,
    then verifies the improvement.
    """

    def __init__(
        self,
        probe: FineProbe,
        surgeon: HeadSurgeon,
        cfg: Optional[CPSurgeonConfig] = None,
    ):
        self.probe = probe
        self.surgeon = surgeon
        self.cfg = cfg or CPSurgeonConfig()
        self.n_layers = probe.n_layers
        self.n_heads = probe.n_heads

    # ── Measurement ────────────────────────────────────────

    def measure_cp(
        self,
        model: torch.nn.Module,
        tokenizer,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """Run prompts through model and compute CP."""
        cp = CausalPrimitives(
            self.n_layers, self.n_heads,
            CPConfig(n_bins=self.cfg.n_bins)
        )

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids)
            self.probe.reset()

            with torch.no_grad():
                try:
                    model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.cfg.max_tokens,
                        do_sample=True,
                        temperature=self.cfg.temperature,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                except Exception:
                    pass

            # Feed diagnostics into CP
            for d in self.probe.diagnostics:
                he = d.head_energies
                le = [sum(heads) for heads in he] if he else []
                cp.observe_token(he, le, d.J)

        return cp.compute()

    # ── Scale computation ──────────────────────────────────

    def compute_target_scales(
        self,
        cp_report: Dict[str, Any],
        reference_cp: Optional[Dict[str, Any]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute per-head scaling factors to improve emergence.

        Strategy: push CP distribution toward the target that maximizes
        S_path × S_row_bar. This means:
          - Across layers: equalize layer CP (maximize S_path)
          - Within layers: differentiate head CP (maximize S_row_bar)

        Practically:
          - Heads with CP far above layer mean → dampen slightly
          - Heads with CP far below layer mean → boost slightly
          - Layers with CP far below model mean → boost all heads in that layer
          - Layers with CP far above model mean → dampen all heads in that layer

        The learning rate controls how aggressive the correction is.
        """
        if cp_report.get("status") != "ok":
            return {}

        head_cp = cp_report["head_cp"]
        layer_cp = cp_report["layer_cp"]
        n_layers = len(head_cp)
        n_heads = len(head_cp[0]) if n_layers > 0 else 0
        lr = self.cfg.learning_rate

        scales = {}

        if self.cfg.target_mode == "restore_early" and reference_cp is not None:
            return self._compute_restore_scales(cp_report, reference_cp)

        # ── Layer-level equalization ──
        # Target: all layers contribute equally to total CP
        total_cp = sum(layer_cp)
        if total_cp < 1e-12:
            return scales

        mean_layer_cp = total_cp / n_layers
        layer_scales = []
        for i in range(n_layers):
            if layer_cp[i] < 1e-12:
                layer_scales.append(1.0)  # Can't scale a dead layer
            else:
                # Ratio of target to current
                ratio = mean_layer_cp / layer_cp[i]
                # Apply learning rate: move partially toward target
                scale = 1.0 + lr * (ratio - 1.0)
                # Clamp
                scale = max(self.cfg.min_dampen, min(self.cfg.max_boost, scale))
                layer_scales.append(scale)

        # ── Within-layer differentiation ──
        # Target: within each layer, reward heads that are doing unique work
        # (high specificity, high determinism) and penalize redundant ones
        for i in range(n_layers):
            layer_head_cps = head_cp[i]
            layer_mean = sum(layer_head_cps) / n_heads if n_heads > 0 else 0
            layer_max = max(layer_head_cps) if layer_head_cps else 0

            for j in range(n_heads):
                hcp = layer_head_cps[j]

                # Start with layer-level scale
                scale = layer_scales[i]

                # Within-layer adjustment
                if layer_max > 1e-12:
                    # How dominant is this head within its layer?
                    dominance = hcp / layer_max

                    if dominance > self.cfg.dominant_threshold:
                        # This head is hogging the layer's causal load
                        # Dampen it to force other heads to step up
                        dampen = 1.0 - lr * 0.3 * (dominance - self.cfg.dominant_threshold)
                        scale *= max(self.cfg.min_dampen, dampen)

                    elif hcp < self.cfg.dead_head_threshold:
                        # Causally dead head — boost it
                        boost = 1.0 + lr * 0.5
                        scale *= min(self.cfg.max_boost, boost)

                # Clamp final scale
                scale = max(self.cfg.min_dampen, min(self.cfg.max_boost, scale))

                # Only record if meaningful change
                if abs(scale - 1.0) > 0.01:
                    scales[(i, j)] = round(scale, 4)

        return scales

    def _compute_restore_scales(
        self,
        current_cp: Dict[str, Any],
        target_cp: Dict[str, Any],
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute scales to restore a reference CP distribution.

        Use this when you have the CP from an early training checkpoint
        (where the model had distributed structure) and want to push
        the current model back toward that distribution.
        """
        if current_cp.get("status") != "ok" or target_cp.get("status") != "ok":
            return {}

        scales = {}
        lr = self.cfg.learning_rate
        c_head = current_cp["head_cp"]
        t_head = target_cp["head_cp"]

        n_layers = min(len(c_head), len(t_head))

        for i in range(n_layers):
            n_heads = min(len(c_head[i]), len(t_head[i]))
            for j in range(n_heads):
                current = c_head[i][j]
                target = t_head[i][j]

                if current < 1e-12:
                    if target > 0.01:
                        scales[(i, j)] = min(self.cfg.max_boost, 1.0 + lr)
                    continue

                ratio = target / current
                scale = 1.0 + lr * (ratio - 1.0)
                scale = max(self.cfg.min_dampen, min(self.cfg.max_boost, scale))

                if abs(scale - 1.0) > 0.01:
                    scales[(i, j)] = round(scale, 4)

        return scales

    # ── Apply ──────────────────────────────────────────────

    def apply_scales(
        self,
        scales: Dict[Tuple[int, int], float],
        reason_prefix: str = "cp_auto",
    ) -> List[HeadIntervention]:
        """Apply computed scales via HeadSurgeon."""
        interventions = []
        for (layer, head), factor in scales.items():
            reason = f"{reason_prefix}: scale={factor:.3f}"
            self.surgeon.scale_head(layer, head, factor, reason)
            interventions.append(HeadIntervention(
                layer=layer, head=head, factor=factor, reason=reason
            ))
        return interventions

    # ── Full auto-tune loop ────────────────────────────────

    def auto_tune(
        self,
        model: torch.nn.Module,
        tokenizer,
        prompts: List[str],
        reference_cp: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> CPSurgeryResult:
        """
        Full auto-tuning loop: measure → compute → apply → verify → iterate.

        Args:
            model: The model to tune
            tokenizer: Tokenizer for encoding prompts
            prompts: List of prompt strings to use for CP measurement
            reference_cp: Optional early-checkpoint CP to restore toward
            verbose: Print progress

        Returns:
            CPSurgeryResult with before/after comparison and convergence history.
        """
        result = CPSurgeryResult()

        if verbose:
            print("CP-Guided Surgery")
            print("=" * 50)

        # ── Initial measurement ──
        if verbose:
            print("Measuring initial CP...", end="", flush=True)

        before = self.measure_cp(model, tokenizer, prompts)
        result.before_cp = before
        result.before_emergence = before.get("emergence", 0)
        result.before_hierarchy = before.get("hierarchy", "?")
        result.before_top_heads = before.get("top_heads", [])[:5]
        result.convergence_history.append(result.before_emergence)

        if verbose:
            print(f" emergence={result.before_emergence:.6f} "
                  f"hierarchy={result.before_hierarchy}")

        if before.get("status") != "ok":
            if verbose:
                print("Insufficient data for CP measurement.")
            return result

        # ── Iterative tuning ──
        current_cp = before
        all_interventions = []

        for iteration in range(self.cfg.max_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}:")

            # Compute target scales
            if self.cfg.target_mode == "restore_early" and reference_cp is not None:
                scales = self._compute_restore_scales(current_cp, reference_cp)
            else:
                scales = self.compute_target_scales(current_cp, reference_cp)

            if not scales:
                if verbose:
                    print("  No adjustments needed.")
                break

            # Apply
            n_boost = sum(1 for s in scales.values() if s > 1.0)
            n_dampen = sum(1 for s in scales.values() if s < 1.0)
            if verbose:
                print(f"  Applying {len(scales)} adjustments "
                      f"({n_boost} boost, {n_dampen} dampen)...", end="", flush=True)

            interventions = self.apply_scales(scales, f"iter_{iteration}")
            all_interventions.extend(interventions)

            # Re-measure
            current_cp = self.measure_cp(model, tokenizer, prompts)
            new_emergence = current_cp.get("emergence", 0)
            result.convergence_history.append(new_emergence)

            if verbose:
                delta = new_emergence - result.convergence_history[-2]
                print(f" emergence={new_emergence:.6f} (Δ={delta:+.6f}) "
                      f"hierarchy={current_cp.get('hierarchy', '?')}")

            # Check convergence
            if abs(new_emergence - result.convergence_history[-2]) < self.cfg.convergence_threshold:
                if verbose:
                    print("  Converged.")
                result.converged = True
                result.iterations = iteration + 1
                break

            result.iterations = iteration + 1

        # ── Final result ──
        result.after_cp = current_cp
        result.after_emergence = current_cp.get("emergence", 0)
        result.after_hierarchy = current_cp.get("hierarchy", "?")
        result.after_top_heads = current_cp.get("top_heads", [])[:5]
        result.interventions = all_interventions

        if result.before_emergence > 0:
            result.emergence_improvement = (
                (result.after_emergence - result.before_emergence)
                / result.before_emergence
            )
        else:
            result.emergence_improvement = 0.0

        # Collect final head scales (cumulative)
        cumulative_scales: Dict[Tuple[int, int], float] = {}
        for iv in all_interventions:
            key = (iv.layer, iv.head)
            cumulative_scales[key] = cumulative_scales.get(key, 1.0) * iv.factor
        result.head_scales = cumulative_scales

        if verbose:
            print(f"\n{result.summary()}")

        return result

    # ── Restore ────────────────────────────────────────────

    def restore(self) -> None:
        """Undo all surgery, restore original weights."""
        self.surgeon.restore()
