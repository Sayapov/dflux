#!/usr/bin/env python3
"""
AXE-NS: Axiomatic Extractor Engine × Navier-Stokes

The full engine. Δ_flux detects. AXE-NS detects, diagnoses, proposes
a fix, verifies the fix preserves structure, and zooms in if things
get worse.

Five towers realized:
    T_S  Sensitivity   — energy monitoring across banded groups
    T_E  Equivalence   — regime classification (laminar/transitional/turbulent)
    T_M  Memory        — witness history with J trajectory
    T_A  Agency        — EXTRACT proposes canonical stabilizing actions
    T_I  Identity      — RENORMALIZE maintains coherence across regime shifts

Usage (PyTorch):
    engine = AXEEngine.from_optimizer(optimizer, L_cut=3)

    for step in range(num_steps):
        loss.backward()
        action = engine.step(step, loss=loss.item(), lr=lr)

        if action.kind == "extract":
            # Engine recommends specific parameter adjustments
            engine.apply_extract(optimizer)
        elif action.kind == "renormalize":
            # Engine recommends regime change (LR reduction, etc.)
            engine.apply_renormalize(optimizer)

        optimizer.step()
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION (K — equivalence classes)
# ══════════════════════════════════════════════════════════════

class Regime(Enum):
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"
    CRITICAL = "critical"


def classify_regime(
    E_tail: float,
    E_total: float,
    flux_ratio: float,
) -> Regime:
    """Reynolds-analogue regime classification.

    Uses tail energy fraction and flux ratio to determine the
    flow regime of gradient dynamics.
    """
    if E_total < 1e-12:
        return Regime.LAMINAR

    tail_fraction = E_tail / E_total

    if tail_fraction < 0.15 and flux_ratio < 0.1:
        return Regime.LAMINAR
    elif tail_fraction < 0.35 or flux_ratio < 0.3:
        return Regime.TRANSITIONAL
    elif tail_fraction < 0.6:
        return Regime.TURBULENT
    else:
        return Regime.CRITICAL


# ══════════════════════════════════════════════════════════════
# J — MONOTONE COMPLEXITY FUNCTIONAL
# ══════════════════════════════════════════════════════════════

@dataclass
class JState:
    """Tracks the complexity functional J and its trajectory.

    Uses EMA (exponential moving average) smoothing to separate
    real trends from batch-to-batch noise. Raw J in transformers
    fluctuates ~50% step-to-step; the EMA reveals the actual
    structural trajectory underneath.
    """
    values: List[float] = field(default_factory=list)
    smoothed: List[float] = field(default_factory=list)
    window: int = 50
    ema_alpha: float = 0.1  # Smoothing factor (lower = smoother)
    monotone_tolerance: float = 0.01  # Default, can be overridden

    def record(self, j: float) -> None:
        self.values.append(j)
        # EMA smoothing
        if not self.smoothed:
            self.smoothed.append(j)
        else:
            s = self.ema_alpha * j + (1 - self.ema_alpha) * self.smoothed[-1]
            self.smoothed.append(s)

    @property
    def current(self) -> Optional[float]:
        return self.smoothed[-1] if self.smoothed else None

    @property
    def current_raw(self) -> Optional[float]:
        return self.values[-1] if self.values else None

    @property
    def previous(self) -> Optional[float]:
        return self.smoothed[-2] if len(self.smoothed) >= 2 else None

    @property
    def monotone_violated(self) -> bool:
        """Smoothed J must not increase beyond tolerance."""
        if len(self.smoothed) < 2:
            return False
        return self.smoothed[-1] > self.smoothed[-2] * (1 + self.monotone_tolerance)

    @property
    def stabilized(self) -> bool:
        """Has smoothed J stopped changing? (attractor detection)"""
        if len(self.smoothed) < self.window:
            return False
        recent = self.smoothed[-self.window:]
        spread = max(recent) - min(recent)
        scale = max(abs(recent[-1]), 1e-12)
        return (spread / scale) < 0.01  # <1% variation over window

    @property
    def trend(self) -> float:
        """Slope of smoothed J over recent window. Negative = good."""
        if len(self.smoothed) < 10:
            return 0.0
        recent = self.smoothed[-min(len(self.smoothed), self.window):]
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / max(den, 1e-12)


# ══════════════════════════════════════════════════════════════
# WITNESS RECORD (W — canonical evidence)
# ══════════════════════════════════════════════════════════════

@dataclass
class Witness:
    step: int
    depth: int
    J: float
    regime: str
    mode: str  # "flowing", "stabilized", "extracted", "renormalized"
    E_tail: float
    E_total: float
    flux: float
    action_kind: str
    details: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# ACTION — what the engine recommends
# ══════════════════════════════════════════════════════════════

@dataclass
class Action:
    kind: str  # "none", "extract", "renormalize", "alert"
    reason: str
    targets: Dict[str, Any] = field(default_factory=dict)
    witness: Optional[Witness] = None


# ══════════════════════════════════════════════════════════════
# AXE-NS CONFIGURATION
# ══════════════════════════════════════════════════════════════

@dataclass
class AXEConfig:
    """Configuration for the AXE-NS engine.

    Args:
        L_cut: Band cutoff. Groups above this are the "tail."
        window_steps: Steps per measurement window.
        theta_warning: Δ_flux threshold for alerts. Set to 0 for auto-calibration.
        beta: Meta-supervision rate (check every β steps).
        extract_decay: LR multiplier when extracting (0 < x < 1).
        renorm_decay: LR multiplier when renormalizing (stronger).
        j_func: Which functional to use for J.
                 "energy" = total gradient energy (default)
                 "tail_ratio" = E_tail / E_total
                 "loss_weighted" = loss × tail_energy
        stagnation_window: Steps to detect stagnation.
        stagnation_eps: Minimum J improvement to not count as stagnant.
        monotone_tolerance: How much smoothed J can increase before violation.
        max_depth: Maximum refinement depth.
        warmup_steps: Steps to observe before making any decisions.
                      During warmup, the engine records baseline statistics
                      and auto-calibrates theta_warning if set to 0.
        ema_alpha: EMA smoothing factor for J (lower = smoother).
        violations_required: Consecutive violations before RENORMALIZE.
    """
    L_cut: int = 0
    window_steps: int = 100
    theta_warning: float = 0.0  # 0 = auto-calibrate from warmup
    beta: int = 50
    extract_decay: float = 0.5
    renorm_decay: float = 0.1
    j_func: str = "energy"
    stagnation_window: int = 100
    stagnation_eps: float = 1e-4
    monotone_tolerance: float = 0.05  # 5% tolerance on smoothed J
    max_depth: int = 5
    positive_only: bool = True
    warmup_steps: int = 50  # Observe before acting
    ema_alpha: float = 0.1  # J smoothing factor
    violations_required: int = 5  # Consecutive violations before RENORMALIZE
    run_id: str = "run-unknown"
    model_id: str = "model-unknown"


# ══════════════════════════════════════════════════════════════
# AXE-NS ENGINE
# ══════════════════════════════════════════════════════════════

class AXEEngine:
    """AXE × Navier-Stokes: the full training stability engine.

    Goes beyond detection (Δ_flux) to diagnosis, prescription,
    and adaptive refinement.
    """

    def __init__(self, n_groups: int, cfg: AXEConfig) -> None:
        self.n_groups = n_groups
        self.cfg = cfg
        self._optimizer = None

        # State
        self.depth = 1                      # Current refinement depth
        self.J = JState(
            window=cfg.stagnation_window,
            ema_alpha=cfg.ema_alpha,
            monotone_tolerance=cfg.monotone_tolerance,
        )
        self.regime = Regime.LAMINAR
        self.regime_history: List[Regime] = []
        self.witnesses: List[Witness] = []

        # Δ_flux internals (T_S layer)
        self._last_E_tail: Optional[float] = None
        self._window_start: Optional[int] = None
        self._window_accum: float = 0.0

        # Band norms history for EXTRACT
        self._norm_history: List[List[float]] = []
        self._norm_window = 20

        # Warmup / auto-calibration
        self._warmup_fluxes: List[float] = []
        self._warmup_j_changes: List[float] = []
        self._calibrated = cfg.theta_warning > 0  # Skip warmup if user set theta

        # Tracking
        self._extractions = 0
        self._renormalizations = 0
        self._violations = 0
        self._consecutive_violations = 0
        self._step_count = 0

        # Output files
        self._events_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None

    # ── Factory ──────────────────────────────────────────────

    @classmethod
    def from_optimizer(
        cls,
        optimizer,
        L_cut: int = 0,
        window_steps: int = 100,
        theta_warning: float = 0.5,
        beta: int = 50,
        csv_path: str = "axe_ns.csv",
        events_path: str = "axe_ns_events.jsonl",
        **kwargs,
    ) -> "AXEEngine":
        n_groups = len(optimizer.param_groups)
        if L_cut == 0:
            L_cut = max(0, int(0.6 * n_groups) - 1)

        cfg = AXEConfig(
            L_cut=L_cut,
            window_steps=window_steps,
            theta_warning=theta_warning,
            beta=beta,
            **kwargs,
        )
        engine = cls(n_groups=n_groups, cfg=cfg)
        engine._optimizer = optimizer
        engine.open(csv_path=csv_path, events_path=events_path)
        return engine

    # ── Lifecycle ────────────────────────────────────────────

    def open(self, csv_path: str, events_path: str) -> None:
        self._events_path = Path(events_path)
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        self._events_path.write_text("", encoding="utf-8")

        csv_p = Path(csv_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)

        header = ["step", "loss", "lr", "J", "J_smoothed", "regime", "depth",
                   "E_tail", "E_total", "flux", "delta_flux_window",
                   "mode", "action"]
        self._csv_file = csv_p.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=header)
        self._csv_writer.writeheader()
        self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None

    # ── Core step ────────────────────────────────────────────

    def step(
        self,
        step: int,
        norms: Optional[List[float]] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> Action:
        """Run one AXE cycle.

        Returns an Action: what the engine recommends doing.
        """
        self._step_count = step

        # ── T_S: Compute norms and energy ──
        if norms is None:
            if self._optimizer is not None:
                norms = self._compute_norms()
            else:
                raise ValueError("Must provide norms= or attach optimizer")

        E_total = sum(n ** 2 for n in norms)
        E_head = sum(n ** 2 for n in norms[:self.cfg.L_cut + 1])
        E_tail = sum(n ** 2 for n in norms[self.cfg.L_cut + 1:])

        # Store norm history for EXTRACT
        self._norm_history.append(list(norms))
        if len(self._norm_history) > self._norm_window:
            self._norm_history.pop(0)

        # ── T_S: Compute flux ──
        if self._last_E_tail is None:
            flux = 0.0
        else:
            dE = E_tail - self._last_E_tail
            flux = max(dE, 0.0) if self.cfg.positive_only else abs(dE)
        self._last_E_tail = E_tail

        # Windowed accumulation
        if self._window_start is None:
            self._window_start = step
            self._window_accum = 0.0
        self._window_accum += flux
        delta_flux_window = self._window_accum

        # Window boundary reset
        if (step - self._window_start + 1) >= self.cfg.window_steps:
            self._window_start = step + 1
            self._window_accum = 0.0

        # ── T_E: Classify regime ──
        flux_ratio = flux / max(E_total, 1e-12)
        self.regime = classify_regime(E_tail, E_total, flux_ratio)
        self.regime_history.append(self.regime)

        # ── Compute J ──
        j_val = self._compute_J(E_tail, E_total, loss)
        self.J.record(j_val)

        # ── Decide action ──
        action = self._decide(step, E_tail, E_total, flux, delta_flux_window, loss, lr)

        # ── T_M: Record witness ──
        witness = Witness(
            step=step,
            depth=self.depth,
            J=j_val,
            regime=self.regime.value,
            mode=action.kind if action.kind != "none" else "flowing",
            E_tail=E_tail,
            E_total=E_total,
            flux=flux,
            action_kind=action.kind,
            details=action.targets,
        )
        self.witnesses.append(witness)
        action.witness = witness

        # ── Emit event if action taken ──
        if action.kind != "none":
            self._emit_event(witness)

        # ── CSV logging ──
        if self._csv_writer:
            self._csv_writer.writerow({
                "step": step,
                "loss": f"{loss:.6f}" if loss is not None else "",
                "lr": f"{lr:.8f}" if lr is not None else "",
                "J": f"{j_val:.6f}",
                "J_smoothed": f"{self.J.current:.6f}" if self.J.current else "",
                "regime": self.regime.value,
                "depth": self.depth,
                "E_tail": f"{E_tail:.6f}",
                "E_total": f"{E_total:.6f}",
                "flux": f"{flux:.6f}",
                "delta_flux_window": f"{delta_flux_window:.6f}",
                "mode": witness.mode,
                "action": action.kind,
            })
            self._csv_file.flush()

        return action

    # ── Decision engine ──────────────────────────────────────

    def _calibrate(self, flux: float) -> None:
        """Collect warmup statistics for auto-calibration."""
        self._warmup_fluxes.append(flux)
        if len(self.J.values) >= 2:
            self._warmup_j_changes.append(
                abs(self.J.values[-1] - self.J.values[-2])
            )

    def _finish_calibration(self) -> None:
        """Set thresholds from warmup statistics."""
        if self._warmup_fluxes:
            # theta = p95 of windowed flux during warmup
            # This means only the top 5% of normal dynamics trigger alerts
            sorted_fluxes = sorted(self._warmup_fluxes)
            p95_idx = min(int(len(sorted_fluxes) * 0.95), len(sorted_fluxes) - 1)
            baseline_flux = sorted_fluxes[p95_idx]
            # Set theta to 3x the p95 windowed flux
            self.cfg.theta_warning = max(baseline_flux * 3.0, 0.1)

        self._calibrated = True

    def _decide(
        self,
        step: int,
        E_tail: float,
        E_total: float,
        flux: float,
        delta_flux: float,
        loss: Optional[float],
        lr: Optional[float],
    ) -> Action:
        """The core AXE decision logic. Five checks in priority order."""

        # ── 0. WARMUP: observe before acting ──
        if step < self.cfg.warmup_steps:
            self._calibrate(flux)
            return Action(kind="none", reason="warmup")

        if not self._calibrated:
            self._finish_calibration()

        # ── 1. MONOTONICITY CHECK (smoothed J must not increase) ──
        if self.J.monotone_violated:
            self._consecutive_violations += 1
            self._violations += 1

            # Need sustained violations, not single-step noise
            if (self._consecutive_violations >= self.cfg.violations_required
                    or self.regime == Regime.CRITICAL):
                self._renormalizations += 1
                self._consecutive_violations = 0
                new_depth = min(self.depth + 1, self.cfg.max_depth)
                self.depth = new_depth

                return Action(
                    kind="renormalize",
                    reason=(f"Smoothed J increasing for {self.cfg.violations_required} steps. "
                            f"Regime: {self.regime.value}. "
                            f"Increasing depth to {new_depth}, "
                            f"reducing LR by {self.cfg.renorm_decay:.0%}"),
                    targets={
                        "lr_multiplier": self.cfg.renorm_decay,
                        "new_depth": new_depth,
                        "J_current": self.J.current,
                        "J_previous": self.J.previous,
                        "regime": self.regime.value,
                    },
                )
        else:
            # Reset consecutive counter on any non-violation
            self._consecutive_violations = 0

        # ── 2. REGIME TRANSITION CHECK ──
        if len(self.regime_history) >= 2:
            prev_regime = self.regime_history[-2]
            if (self.regime == Regime.TURBULENT and
                    prev_regime in (Regime.LAMINAR, Regime.TRANSITIONAL)):
                self._extractions += 1
                return Action(
                    kind="extract",
                    reason=(f"Regime transition: {prev_regime.value} → "
                            f"{self.regime.value}. Extracting canonical state."),
                    targets={
                        "lr_multiplier": self.cfg.extract_decay,
                        "from_regime": prev_regime.value,
                        "to_regime": self.regime.value,
                        "E_tail": E_tail,
                        "E_total": E_total,
                    },
                )

            if self.regime == Regime.CRITICAL:
                self._extractions += 1
                return Action(
                    kind="extract",
                    reason=f"Critical regime reached. E_tail/E_total = {E_tail/max(E_total,1e-12):.2%}",
                    targets={
                        "lr_multiplier": self.cfg.extract_decay,
                        "regime": self.regime.value,
                        "E_tail_fraction": E_tail / max(E_total, 1e-12),
                    },
                )

        # ── 3. META-SUPERVISION (every β steps) ──
        if step > 0 and step % self.cfg.beta == 0:
            if self.J.trend > self.cfg.stagnation_eps:
                self._extractions += 1
                return Action(
                    kind="extract",
                    reason=f"J trending upward (slope={self.J.trend:.6f}) at β-check",
                    targets={
                        "lr_multiplier": self.cfg.extract_decay,
                        "J_trend": self.J.trend,
                    },
                )

        # ── 4. STABILIZATION CHECK ──
        if self.J.stabilized:
            return Action(
                kind="stabilized",
                reason=f"J stabilized (<1% variation over {self.J.window} steps)",
                targets={"J": self.J.current, "regime": self.regime.value},
            )

        # ── 5. FLUX THRESHOLD (Δ_flux alert, same as before) ──
        if delta_flux > self.cfg.theta_warning:
            return Action(
                kind="alert",
                reason=f"Δ_flux={delta_flux:.2f} > θ={self.cfg.theta_warning:.2f}",
                targets={
                    "delta_flux": delta_flux,
                    "theta": self.cfg.theta_warning,
                    "regime": self.regime.value,
                },
            )

        return Action(kind="none", reason="flowing")

    # ── T_A: Apply actions ───────────────────────────────────

    def apply_extract(self, optimizer) -> Dict[str, Any]:
        """EXTRACT: Find J-minimal configuration in equivalence class.

        In practice: reduce LR for tail groups (deep layers) to
        their canonical stable rate based on recent norm history.
        Preserves the regime (equivalence class) while minimizing J.
        """
        multiplier = self.cfg.extract_decay
        changes = {}

        for i, pg in enumerate(optimizer.param_groups):
            if i > self.cfg.L_cut:
                # Tail group — apply stronger decay
                old_lr = pg["lr"]
                pg["lr"] = old_lr * multiplier
                changes[f"group_{i}_lr"] = {"old": old_lr, "new": pg["lr"]}
            else:
                # Head group — lighter touch
                old_lr = pg["lr"]
                pg["lr"] = old_lr * (1 - (1 - multiplier) * 0.3)
                changes[f"group_{i}_lr"] = {"old": old_lr, "new": pg["lr"]}

        return changes

    def apply_renormalize(self, optimizer) -> Dict[str, Any]:
        """RENORMALIZE: Regime transition. Increase resolution, reduce all LRs.

        This is the "zoom in" operation — we increase depth (finer
        band resolution in future steps) and aggressively reduce LR
        to find the new stable operating point.
        """
        multiplier = self.cfg.renorm_decay
        changes = {"depth": self.depth}

        for i, pg in enumerate(optimizer.param_groups):
            old_lr = pg["lr"]
            pg["lr"] = old_lr * multiplier
            changes[f"group_{i}_lr"] = {"old": old_lr, "new": pg["lr"]}

        # Also increase weight decay slightly to strengthen T_I
        for i, pg in enumerate(optimizer.param_groups):
            old_wd = pg.get("weight_decay", 0)
            pg["weight_decay"] = old_wd * 1.5 + 1e-5
            changes[f"group_{i}_wd"] = {"old": old_wd, "new": pg["weight_decay"]}

        return changes

    # ── Internals ────────────────────────────────────────────

    def _compute_J(self, E_tail: float, E_total: float, loss: Optional[float]) -> float:
        """Compute the complexity functional J."""
        if self.cfg.j_func == "energy":
            return E_total
        elif self.cfg.j_func == "tail_ratio":
            return E_tail / max(E_total, 1e-12)
        elif self.cfg.j_func == "loss_weighted":
            return (loss or 0.0) * (1 + E_tail / max(E_total, 1e-12))
        else:
            return E_total

    def _compute_norms(self) -> List[float]:
        """Compute gradient L2 norms from optimizer."""
        import torch
        norms = []
        for group in self._optimizer.param_groups:
            sq = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                sq += float(torch.sum(g * g).cpu())
            norms.append(math.sqrt(max(sq, 0.0)))
        return norms

    def _emit_event(self, w: Witness) -> None:
        """Write witness to JSONL."""
        if self._events_path:
            event = {
                "schema_version": "2.0",
                "engine": "axe_ns",
                "run_id": self.cfg.run_id,
                "model_id": self.cfg.model_id,
                "step": w.step,
                "depth": w.depth,
                "J": w.J,
                "regime": w.regime,
                "mode": w.mode,
                "action": w.action_kind,
                "E_tail": w.E_tail,
                "E_total": w.E_total,
                "flux": w.flux,
                "details": w.details,
            }
            with self._events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    # ── Summary ──────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return engine statistics."""
        regime_counts = {}
        for r in self.regime_history:
            regime_counts[r.value] = regime_counts.get(r.value, 0) + 1

        return {
            "total_steps": self._step_count,
            "current_depth": self.depth,
            "current_regime": self.regime.value,
            "extractions": self._extractions,
            "renormalizations": self._renormalizations,
            "monotone_violations": self._violations,
            "J_final_smoothed": self.J.current,
            "J_final_raw": self.J.current_raw,
            "J_stabilized": self.J.stabilized,
            "J_trend": self.J.trend,
            "theta_warning": self.cfg.theta_warning,
            "warmup_steps": self.cfg.warmup_steps,
            "auto_calibrated": not (self.cfg.theta_warning > 0 and len(self._warmup_fluxes) == 0),
            "regime_distribution": regime_counts,
            "witness_count": len(self.witnesses),
        }
