#!/usr/bin/env python3
"""Δ_flux Meter — core instrument.

Watches energy distribution across banded signal groups.
When windowed instability (Δ_flux) crosses a threshold, emits one JSONL event.

Framework-agnostic core. The only requirement is that you can provide
a list of L2 norms per group at each step.

For PyTorch optimizer integration, use DFluxMeter.from_optimizer().
For HuggingFace Trainer, see dflux.adapters.hf_trainer.
For PyTorch Lightning, see dflux.adapters.lightning.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"
EVENT_TYPE_THRESHOLD = "dflux.threshold_exceeded"


@dataclass
class DFluxConfig:
    """Configuration for the Δ_flux meter.

    Args:
        L_cut: Band cutoff index. Groups above this index are the "tail."
        window_steps: How many steps per measurement window.
        theta_warning: Δ_flux threshold for warning events.
        theta_critical: Δ_flux threshold for critical events. Default: 2x warning.
        log_every: Write CSV row every N steps. Default: 1.
        positive_only: Only count positive flux (energy flowing INTO tail). Default: True.
        run_id: Identifier for this training run.
        model_id: Identifier for the model being trained.
        framework: Framework name (pytorch, jax, tf, other).
        band_type: What the bands represent (param_group, layer, channel, etc.).
        time_unit: What each step represents (step, epoch, second, etc.).
    """
    L_cut: int = 0
    window_steps: int = 200
    theta_warning: float = 0.5
    theta_critical: Optional[float] = None

    log_every: int = 1
    positive_only: bool = True

    run_id: str = "run-unknown"
    model_id: str = "model-unknown"
    framework: str = "pytorch"

    band_type: str = "param_group"
    time_unit: str = "step"

    def __post_init__(self):
        if self.theta_critical is None:
            self.theta_critical = 2.0 * self.theta_warning


class DFluxMeter:
    """Core Δ_flux instrument.

    Usage (framework-agnostic):
        meter = DFluxMeter(n_groups=10, cfg=DFluxConfig(L_cut=5))
        meter.open(csv_path="dflux.csv", events_path="dflux_events.jsonl")

        for step in range(num_steps):
            norms = [compute_norm(group) for group in groups]
            result = meter.step(step, norms=norms)

        meter.close()

    Usage (PyTorch):
        meter = DFluxMeter.from_optimizer(optimizer, L_cut=5, window_steps=200, theta_warning=0.5)
        # ... call meter.step(step) after loss.backward(), before optimizer.step()

    Usage (HuggingFace Trainer):
        from dflux.adapters.hf_trainer import DFluxCallback
        trainer = Trainer(..., callbacks=[DFluxCallback()])

    Usage (PyTorch Lightning):
        from dflux.adapters.lightning import DFluxCallback
        trainer = pl.Trainer(callbacks=[DFluxCallback()])
    """

    def __init__(self, n_groups: int, cfg: DFluxConfig) -> None:
        self.n_groups = n_groups
        self.cfg = cfg

        self._csv_file = None
        self._csv_writer = None
        self._events_path: Optional[Path] = None

        # Internal state
        self._last_E_tail: Optional[float] = None
        self._window_start_step: Optional[int] = None
        self._window_accum: float = 0.0

        # Optional PyTorch optimizer reference
        self._optimizer = None

    # ── Factory: from PyTorch optimizer ──────────────────────

    @classmethod
    def from_optimizer(
        cls,
        optimizer,
        L_cut: int,
        window_steps: int = 200,
        theta_warning: float = 0.5,
        theta_critical: Optional[float] = None,
        positive_only: bool = True,
        log_every: int = 1,
        csv_path: str = "dflux.csv",
        events_path: str = "dflux_events.jsonl",
        run_id: str = "run-unknown",
        model_id: str = "model-unknown",
        framework: str = "pytorch",
    ) -> "DFluxMeter":
        """Create a meter attached to a PyTorch optimizer.

        Automatically determines n_groups from optimizer.param_groups
        and computes gradient norms on each step() call.
        """
        n_groups = len(optimizer.param_groups)
        cfg = DFluxConfig(
            L_cut=L_cut,
            window_steps=window_steps,
            theta_warning=theta_warning,
            theta_critical=theta_critical,
            positive_only=positive_only,
            log_every=log_every,
            run_id=run_id,
            model_id=model_id,
            framework=framework,
        )
        meter = cls(n_groups=n_groups, cfg=cfg)
        meter._optimizer = optimizer
        meter.open(csv_path=csv_path, events_path=events_path)
        return meter

    # ── Lifecycle ────────────────────────────────────────────

    def open(
        self,
        csv_path: str = "dflux.csv",
        events_path: str = "dflux_events.jsonl",
        include_loss_lr: bool = True,
    ) -> None:
        """Open CSV and JSONL output files."""
        self._include_loss_lr = include_loss_lr
        self._csv_path = Path(csv_path)
        self._events_path = Path(events_path)

        # CSV header
        header = ["step"]
        if include_loss_lr:
            header += ["loss", "lr"]
        header += [f"norm_G{i}" for i in range(self.n_groups)]
        header += ["E_tail", "Pi_L", "delta_flux_window"]

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=header)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        self._events_path.write_text("", encoding="utf-8")

        # Reset state
        self._last_E_tail = None
        self._window_start_step = None
        self._window_accum = 0.0

    def close(self) -> None:
        """Flush and close output files."""
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ── Core step ────────────────────────────────────────────

    def step(
        self,
        step: int,
        norms: Optional[List[float]] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record one step.

        Args:
            step: Global step number.
            norms: L2 norms per group. If None and a PyTorch optimizer is
                   attached, norms are computed from optimizer.param_groups.
            loss: Optional loss value for context logging.
            lr: Optional learning rate for context logging.

        Returns:
            Dict with step, E_tail, Pi_L, delta_flux_window, event_emitted.
        """
        if norms is None:
            if self._optimizer is not None:
                norms = self._compute_optimizer_norms()
            else:
                raise ValueError(
                    "Must provide norms= or attach a PyTorch optimizer via from_optimizer()"
                )

        if len(norms) != self.n_groups:
            raise ValueError(
                f"Expected {self.n_groups} norms, got {len(norms)}"
            )

        # ── Compute tail energy ──
        if self.cfg.L_cut >= self.n_groups - 1:
            E_tail = 0.0
        else:
            E_tail = sum(n ** 2 for n in norms[self.cfg.L_cut + 1:])

        # ── Compute flux ──
        if self._last_E_tail is None:
            Pi = 0.0
        else:
            dE = E_tail - self._last_E_tail
            Pi = max(dE, 0.0) if self.cfg.positive_only else abs(dE)

        self._last_E_tail = E_tail

        # ── Windowed accumulation ──
        if self._window_start_step is None:
            self._window_start_step = step
            self._window_accum = 0.0

        self._window_accum += Pi
        delta_flux_window = self._window_accum

        # ── Threshold check at window boundary ──
        emitted = False
        if (step - self._window_start_step + 1) >= self.cfg.window_steps:
            if delta_flux_window > self.cfg.theta_warning:
                self._emit_event(
                    step0=self._window_start_step,
                    step1=step,
                    delta_flux=delta_flux_window,
                    loss=loss,
                    lr=lr,
                )
                emitted = True
            self._window_start_step = step + 1
            self._window_accum = 0.0

        # ── CSV logging ──
        if self._csv_writer and (step % self.cfg.log_every == 0):
            row: Dict[str, Any] = {"step": step}
            if self._include_loss_lr:
                row["loss"] = None if loss is None else float(loss)
                row["lr"] = None if lr is None else float(lr)
            for i, v in enumerate(norms):
                row[f"norm_G{i}"] = float(v)
            row["E_tail"] = float(E_tail)
            row["Pi_L"] = float(Pi)
            row["delta_flux_window"] = float(delta_flux_window)
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        return {
            "step": step,
            "E_tail": float(E_tail),
            "Pi_L": float(Pi),
            "delta_flux_window": float(delta_flux_window),
            "event_emitted": emitted,
        }

    # ── Internal ─────────────────────────────────────────────

    def _compute_optimizer_norms(self) -> List[float]:
        """Compute gradient L2 norms from PyTorch optimizer param_groups."""
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

    def _emit_event(
        self,
        *,
        step0: int,
        step1: int,
        delta_flux: float,
        loss: Optional[float],
        lr: Optional[float],
    ) -> None:
        """Write one JSONL event."""
        severity = "critical" if delta_flux > self.cfg.theta_critical else "warning"

        event = {
            "schema_version": SCHEMA_VERSION,
            "event_type": EVENT_TYPE_THRESHOLD,
            "run_id": self.cfg.run_id,
            "model_id": self.cfg.model_id,
            "framework": self.cfg.framework,
            "time_unit": self.cfg.time_unit,
            "t_start": step0,
            "t_end": step1,
            "band_type": self.cfg.band_type,
            "band_cut": self.cfg.L_cut,
            "band_total": self.n_groups,
            "delta_flux": delta_flux,
            "threshold": self.cfg.theta_warning if severity == "warning" else self.cfg.theta_critical,
            "severity": severity,
            "summary": "Sustained energy surge in deep bands",
            "context": {
                "loss": None if loss is None else float(loss),
                "learning_rate": None if lr is None else float(lr),
            },
        }

        if self._events_path:
            with self._events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
